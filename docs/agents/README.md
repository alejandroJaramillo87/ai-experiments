# Agents

How agents work in this repository: the tooling under `.claude/`, the roles, commands and
skills it defines, the escalation ladder that says which to reach for, and the workflows
for running a session. OpenCode and Claude Code are kept at parity, so everything here
applies to both unless a page says otherwise.

An agent session is contained: no host Docker daemon, no GPU, no running services, no
`/mnt/ai-data`. `AGENTS.md` owns that rule and what follows from it; every `make agent-*`,
`make worktree-*`, service and benchmark target in these pages is a host-side command the
human runs, and the reported output is the evidence.

## The ladder

**Start every piece of work at the lowest rung that could plausibly succeed.** Escalate
only when a rung fails. Rungs are reasoning-effort levels on the single frontier model;
there is no second, cheaper model ([tooling-parity.md](tooling-parity.md) has the model
policy).

| Rung | Use when | How | Effort |
|------|----------|-----|--------|
| 1. Batch | Mechanical and fully specified: pin audits, doc-drift fixes, formatting sweeps | `make agent-run` on the host (Claude Code only), or `make agent-sandbox AGENT=opencode PROMPT=...` contained | low |
| 2. Roadmap task | A defined roadmap item with files, steps and an acceptance check | `/task <roadmap> <id>`; the built-in `Explore` and the `skeptic` role | medium |
| 3. Design or debugging | Open-ended work that resists rungs 1 and 2 | An interactive session, human in the loop | high |
| 4. Quality campaign | A subsystem needs auditing, not a task doing | `/bugbash <scope>`; the QA roles, in batches | high or max |

Rung 1 is non-interactive and auto-approves everything not denied. The permission deny
rules in `.claude/settings.json` and `opencode.json` and the `guard.sh` hook are the
safety floor and fire regardless of permission mode; review them before the first
unattended run. `make agent-run` refuses `AGENT=opencode` because OpenCode batch mode
auto-approves every action not explicitly denied, so an OpenCode batch runs contained.

## Roles

Rendered from `.claude/agents/`. A role's tools say what it can touch; the QA roles that
cannot write are kept that way on purpose ([roles.md](roles.md)).

<!-- inventory:roles:start -->

| Role | Tools | Does |
|------|-------|------|
| `bench-runner` | Read, Grep, Glob, Bash | Runs the repo's performance benchmarks and reports reproducible, controlled numbers. |
| `bug-hunter` | Read, Grep, Glob, Bash | Defect finder. |
| `doc-auditor` | Read, Grep, Glob, Bash | Documentation drift detector. |
| `doc-writer` | Read, Grep, Glob, Edit, Write, Bash | Writes and updates documentation following the .styles guides. |
| `feature-implementer` | Read, Grep, Glob, Edit, Write, Bash | Implements a new feature or capability from a stated requirement, style-guide first, with a test that expresses the requirement. |
| `integrator` | Read, Grep, Glob, Bash | Batch gatekeeper. |
| `perf-auditor` | Read, Grep, Glob, Bash | Runtime and memory auditor for repo-local code. |
| `remediator` | Read, Grep, Glob, Edit, Write, Bash | Defect remediation. |
| `skeptic` | Read, Grep, Glob, Bash | Adversarial verifier. |
| `triage-agent` | Read, Grep, Glob | Findings triage. |

<!-- inventory:roles:end -->

## Commands

Rendered from `.claude/commands/`, which `.opencode/commands/` mirrors by symlink.

<!-- inventory:commands:start -->

| Command | Does |
|---------|------|
| `/audit-docs` | Sweep a docs subtree against the Dockerfile/compose truth and return a drift list. |
| `/bench` | Check preconditions, run the benchmark suite for a service, and return the results table. |
| `/bugbash` | Run a bounded quality campaign over a scope — discover, verify, triage, remediate in batches, and push each batch. |
| `/pins` | Audit docker/ and every compose file for unpinned image tags and clones (:latest, :main, floating refs). |
| `/postmortem` | Record a research-sprint postmortem (tried / learned / continue-or-pivot) and append it to the research log. |
| `/task` | Drive one roadmap task through implement -> acceptance-check -> skeptic, then stop for review. |
| `/verify-host` | Verify a claim that needs the workstation — state acceptance criteria, hand over an exact command block, stop, then judge the reported output. |

<!-- inventory:commands:end -->

## Skills

Rendered from `.claude/skills/`, the one location both tools read.

<!-- inventory:skills:start -->

| Skill | Does |
|-------|------|
| `agent-swarm` | Mechanics contract for running a multi-agent workflow that survives to completion - the durable-ledger rule, the fan-in partition rule, the output-budget rule, and the recovery procedure for a run that dies mid-flight. |
| `benchmarking` | The reproducible benchmark protocol for this workstation — which entry points to run, warm-up policy, how results are recorded and compared, and what counts as a regression. |
| `bug-bash` | The method contract for a quality campaign — the phase ladder, the standing dimension list, the severity rubric, what counts as a confirmed finding, and the batch-and-push cadence. |
| `docker-debug` | Container debugging runbook for this inference stack — per-service healthcheck endpoints, common CUDA/driver failure signatures, log locations, and the safe restart procedure. |
| `training-runs` | How to launch, resume, and serve model training runs on this stack — checkpoint layout, stepping a recipe on the GPU, and the train-and-serve round trip. |
| `verify-host` | The handshake for verifying anything that needs the workstation — how to state acceptance criteria before running, hand the human an exact command block, and record the reported output as evidence. |

<!-- inventory:skills:end -->

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [roles.md](roles.md) | What each role in `.claude/agents/` is for, why the QA roles are shaped as they are, the two settings files that decide what any of them may touch, and why host verification is a skill rather than a role. |
| [tooling-parity.md](tooling-parity.md) | This repo is usable from both OpenCode and Claude Code without content drift. |
| [workflows.md](workflows.md) | The recipes for running an agent: a host batch run, a contained session and the secrets it needs, the shared docker footprint, the worktree lifecycle, session export, the maintenance targets, and what to do when one of them refuses. |

<!-- inventory:pages:end -->

## Keeping the definitions honest

Once a month, check both tools' changelogs for native usage reporting, skill or hook API
changes, and any custom piece here that upstream now provides. Delete what upstream
provides; do not maintain a private copy of a feature the tool ships. The precedent is in
[tooling-parity.md](tooling-parity.md): Claude Code's built-in `/usage` and `Explore`
retired the usage plugin on that side and the `explorer` role outright.

The three tables above are generated, so a definition cannot exist without appearing
here and cannot be deleted without the gate noticing. A name whose only other hit in the
repository is its own definition file is a candidate for removal: a definition nothing
invokes is not free, it still has to be kept true.

## Related documentation

- [../../AGENTS.md](../../AGENTS.md) — the operating rules an agent works under, and the router to the rest
- [../sandbox/agent-sandbox.md](../sandbox/agent-sandbox.md) — the container image, its mounts and the egress allowlist
- [../../scripts/agents/README.md](../../scripts/agents/README.md) — the hooks and checks as scripts
- [../../tests/agents/README.md](../../tests/agents/README.md) — the deterministic tests and the role acceptance tests
