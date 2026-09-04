# Agent SDLC: the escalation ladder

The standing workflow for doing work in this repo with OpenCode or Claude Code. The tools are
kept at parity (see `docs/agents/tooling-parity.md`); this doc is the once-documented ladder plus a
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
second, cheaper model (see the model policy in `docs/agents/tooling-parity.md`).

| Rung | Use when | How | Effort |
|------|----------|-----|--------|
| 1. Batch | Mechanical & fully specified: pin audits, doc-drift fixes, formatting sweeps | `make agent-run` (`opencode run --auto` / `claude -p`) | low |
| 2. Roadmap task | A defined roadmap item with files/steps/acceptance | `/task <roadmap> <id>`; built-in `Explore` + skeptic subagents | medium |
| 3. Design / debugging | Open-ended work that resists rungs 1–2 | Interactive frontier session, human in the loop | high |
| 4. Quality campaign | A subsystem needs auditing, not a task doing | `/bugbash <scope>`; the QA roles, in batches | high / max |

Rung 1 is non-interactive and auto-approves everything not denied; the Phase 1 permission deny
rules and the Phase 4 `guard.sh` PreToolUse hook are the safety floor and fire regardless of
permission mode. Review the deny rules (`.claude/settings.json`, `opencode.json`) before the
first unattended run.

### The two settings files

The effective Claude Code policy is the union of `.claude/settings.json` and
`.claude/settings.local.json`. Only the first is version-controlled. The local file is git-ignored, so it is per-checkout
state that never reaches a review, never reaches a branch, and is destroyed with the worktree
it grew in. `scripts/agents/preflight-scan.sh` inspects it anyway, because being local is not
the same as being trusted.

The division follows from that:

- `settings.json` holds standing policy: every `deny`, every `ask`, and the allows that
  describe how work is done in this repository regardless of who is doing it. Anything that
  should survive a worktree, or that anyone else should inherit, belongs here.
- `settings.local.json` holds only what is specific to one checkout or one image — in a
  sandbox worktree, currently the absolute `/opt/pyenv/bin/python3` path.

An allow accretes there whenever a prompt is approved with "don't ask again", so the file
drifts toward being policy that nobody decided and that nobody else can see. Two checkouts of
this repository will hold different files, each shaped by what happened to be inconvenient
that day. Re-read it periodically and empty it: promote what is genuinely policy into
`settings.json`, and delete the rest rather than carrying it. `tests/agents/test_settings.py`
asserts the two do not overlap and that the key-material denies survive whatever the allow
list grows into; it treats an absent local file as normal, because on a fresh clone that is
the correct state.

### Rung 4: the quality campaign

Rungs 1–3 do a task. Rung 4 audits a subsystem that nobody has a task for — the shape of
work where the deliverable is a findings ledger and a set of fixes, not a feature.

It costs orders of magnitude more than rung 3 and takes hours, so escalate to it
deliberately: when a subsystem feeds something else that would silently inherit its
defects, before trusting a component on real data for the first time, or after a large
merge. Do not reach for it because a bug is hard to find.

`/bugbash <scope>` drives the phase ladder. Six roles carry it — `bug-hunter` discovers,
`skeptic` verifies, `triage-agent` packages, `remediator` fixes, `integrator` gates each
batch, and `perf-auditor` and `doc-auditor` own the two dimensions with their own
measurement rules. They share one definition of the terms in
`.claude/skills/bug-bash/SKILL.md`; read it before running one.

Two properties of the cadence matter more than the roles. Discovery is persisted into a
campaign log under `docs/verification/` and pushed **before** any budget goes to fixes,
and every remediation batch is pushed before the next begins — so an interrupted campaign
loses one batch rather than everything. Campaign 2 was stopped mid-flight by a usage limit
and survived only because the ledger had already landed.

### Feature work and host verification

Two roles sit outside the campaign ladder.

`feature-implementer` takes a stated requirement to working code. It is deliberately distinct
from `remediator`: a remediator closes a verified finding and must show the defect existed by
watching a test fail first, whereas feature work has no prior red to reproduce. Its discipline
comes instead from reading the governing `.styles/` guide before writing, expressing the
requirement as a test, and reporting the gate's skip count rather than its exit code. It runs
the Rust gate in the sandbox, because the image carries the pinned toolchain, and it is
required to name the parts its run did not exercise rather than imply full coverage.

Host verification is a **skill, not an agent**. `/verify-host <claim>` loads
`.claude/skills/verify-host/SKILL.md` in the main conversation. The reason is structural: a
subagent has no channel to the human, so it cannot emit a command, pause, and read the reply.
Delegating host verification to a subagent leaves it able only to report failure or to invent
output, and inventing output is the one thing this repository forbids most emphatically.

The skill encodes the handshake — state the acceptance criteria before producing any command,
hand over one copy-pasteable block, stop, then judge each criterion against the verbatim
reported output. It also carries the failure mode this repository keeps hitting: a worktree is
cut from the `HEAD` commit, so a verification run against uncommitted work exercises stale
source and reports a false pass.

## Running agents: user guide

Step-by-step recipes for the `make` workflows. Every target below prints `[ERROR]` with a
solution when a required variable or secret is missing.

### Host batch run (rung 1, Claude Code only)

```bash
make agent-run PROMPT="Audit docker/ for unpinned image tags" EFFORT=low ALLOW='Read Grep'
```

Runs `claude -p` on the host at the given reasoning effort and appends token usage to
`.agent-state/usage.jsonl`. Only Claude Code runs on the host: `AGENT=opencode` refuses by
design, because OpenCode batch mode (`opencode run --auto`) auto-approves every action not
explicitly denied. The refusal message redirects to the sandbox command below.

### Sandboxed session (either tool)

```bash
make agent-sandbox AGENT=claude NAME=mytask                  # interactive
make agent-sandbox AGENT=opencode NAME=mytask PROMPT="..."   # batch, contained
```

Creates the worktree `../ai-experiments-<NAME>` automatically if it does not exist, then
starts a contained session (network-restricted container, egress proxy) in that worktree.
Each tool receives exactly one secret:

- `AGENT=claude` requires `CLAUDE_CODE_OAUTH_TOKEN` — generate once with
  `claude setup-token` on the host, then either export it or save it to the auto-loaded
  secrets file (below).
- `AGENT=opencode` requires an endpoint-scoped `RUNPOD_API_KEY`.

To start every container authenticated without a per-shell export, save the token once to a
mode-600 file outside the repository that `make agent-sandbox` reads on the host. Generate the
token interactively first — `claude setup-token` opens a browser flow and prints the token. Do
**not** wrap it in `$(...)`: it is an interactive command, and a captured value is the on-screen
UI text, not a token.

```bash
claude setup-token            # complete the browser flow, then copy the printed token
mkdir -p ~/.config/ai-experiments
printf 'CLAUDE_CODE_OAUTH_TOKEN=%s\n' 'PASTE_TOKEN_HERE' \
    > ~/.config/ai-experiments/agent-secrets.env    # paste the token over the placeholder
chmod 600 ~/.config/ai-experiments/agent-secrets.env
```

The file must contain only unquoted `KEY=value` lines. An exported `CLAUDE_CODE_OAUTH_TOKEN`
still takes precedence; the file only fills in a secret not already in the environment, and a
malformed file is reported rather than executed. The same file may hold `RUNPOD_API_KEY` for
OpenCode sessions.

`CLAUDE_THEME` (default `dark`) and `EFFORT` (default `low`) preseed the Claude Code
theme and default reasoning effort each session, because the sandbox home is ephemeral
tmpfs. See the persistent settings section in
[agent-sandbox.md](../sandbox/agent-sandbox.md).

### Sandbox docker resources

The sandbox's docker footprint is shared across every worktree, not per-worktree. The make
wrapper pins the compose project to `ai-agents`, so every session — from any worktree — uses one
proxy container (`agent-proxy`), the `ai-agents_agent-internal` and `ai-agents_agent-egress`
networks, and two built images. Pinning is required: the `agent-internal` subnet is fixed at
`172.31.0.0/16`, so an unpinned per-worktree project name collides with `Pool overlaps with
other one on this address space`. The per-session container is `--rm` and named `agent-<NAME>`
while it runs, so `docker ps` maps a live session to its worktree `../ai-experiments-<NAME>`.

`make agent-sandbox` runs `docker compose run` with no `--build`, so a change to the proxy
configuration — most often the egress allowlist in `docker/agents/proxy/squid.conf` — has no
effect until the image is rebuilt:

```bash
docker compose -f docker-compose.agents.yaml -p ai-agents build agent-proxy
```

An allowlist edit that appears to change nothing indicates a stale image rather than a wrong
entry. A domain the proxy does not carry surfaces as a transport error such as
`Socket is closed`, not as a permission message, so check the allowlist before diagnosing a
network fault: `docker exec agent-proxy cat /var/log/squid/access.log`.

Tear the shared footprint down with `make agent-sandbox-clean`. It runs `docker compose -f
docker-compose.agents.yaml -p ai-agents down --remove-orphans`, then sweeps any leftover
`*_agent-internal`/`*_agent-egress` networks from older, unpinned runs; it never touches the
inference stack. Add `--rmi local` to that compose command to also drop the built images. If a
crashed session leaves an `agent-<NAME>` container behind, `make agent-sandbox-clean` removes it
(or `docker rm agent-<NAME>`).

`make clean` is scoped to the inference stack (`docker-compose.inference.yaml`) and no longer runs a
host-wide `docker system prune -f`; that global sweep now lives in the explicit `make prune`
target.

### Worktree lifecycle

```bash
make worktree-mytask           # create ../ai-experiments-mytask on branch task/mytask
make worktree-mytask TIER=full # ... with the full environment (torch, training, GPU)
make worktree-ls               # list worktrees
make worktree-clean mytask     # remove when merged
```

A new worktree starts from the current `HEAD` **commit**. Uncommitted changes in the
current worktree do not reach it, so commit or stash before creating one. With `TIER=full`
this mistake stays invisible until after a multi-minute `flash-attn` build, and then
presents as a test suite failing against code that appears fixed.

Creating a worktree provisions its `.venv` automatically. `TIER` selects how much:

| `TIER` | Sync command | Use for |
|--------|--------------|---------|
| `sandbox` (default) | `uv sync --frozen --only-group dev --only-group sandbox` | Agent sessions and non-GPU work. Torch-free, provisions in seconds. |
| `full` | `uv sync --frozen --all-groups --all-extras` | The full test suite, training, and GPU work. Matches `make install`. |

The `sandbox` tier runs `make quality` and the non-GPU `make test` suite without a manual
sync. It does not install torch, so the torch-gated tests **skip rather than fail**.
`tests/conftest.py` reports which happened beside the pytest summary: `[TIER] sandbox:`
means torch was absent and the run is partial, `[TIER] full:` means the whole suite was in
scope. Read that line, and the skip count, before treating a green suite as a full result.

The `full` tier compiles CUDA extensions (`flash-attn` builds from an sdist against the
cu130 torch pin), so expect several GB of disk and a long build. `MAX_JOBS` and
`NVCC_THREADS` control build parallelism: `make worktree-mytask TIER=full MAX_JOBS=8`.

An unrecognised `TIER` fails with a styled `[ERROR]` before the worktree is created.
Provisioning itself is best-effort: if it cannot resolve (a stale `uv.lock`, no network,
or `uv` absent), the target prints a `[WARN]` with the manual `make install` command and
worktree creation still succeeds.

A bare `uv sync` repairs a worktree environment safely: `[tool.uv] default-groups` in
`pyproject.toml` names `dev`, `sandbox`, `interp`, and `training`, so the reflexive command
adds what is missing instead of removing the tiers it was not told about. It installs torch
into a `sandbox` tier worktree, which is the one cost; `make install` is the fuller form,
adding the `quant` extra.

Removal accepts the bare name, the `task/<name>` branch, a worktree path, `NAME=<name>`,
or the suffix form `make worktree-clean-<name>` — all resolve to the same worktree. A bare
`make worktree-clean` (no name) errors with usage instead of matching the creation
pattern.

Removal reports each outcome separately. Removing a worktree that is already gone prunes
the stale registration, reports `[SKIP]`, and exits 0, so repeating the command is
harmless. Removal from inside the target worktree is refused, because it would leave the
shell without a working directory. A worktree holding uncommitted changes is also refused,
with those changes listed; `FORCE=1` removes it anyway and discards them permanently.

`make worktree-clean` removes only the git worktree; the sandbox's docker resources
are shared, not per-worktree, and are torn down separately with `make agent-sandbox-clean`
(see the Sandbox docker resources note above). Delete the task branch after removal:
`git branch -D task/mytask`. If the branch is kept, recreating the worktree with
`make worktree-mytask` reuses it.

### Session export

```bash
make agent-export AGENT=claude ID=<session-id>
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
| `make agent-role-tests [ROLE=x]` | LLM acceptance test per subagent role (spends tokens) |
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
- `[ERROR] CLAUDE_CODE_OAUTH_TOKEN not set`: run `claude setup-token` on the host, then either
  export the token or save it to `~/.config/ai-experiments/agent-secrets.env` (chmod 600) so
  `make agent-sandbox AGENT=claude` loads it automatically.
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

Precedent already set, and acted on: **Claude Code subsumed usage tracking** (built-in
`/usage`, so `usage.ts` is OpenCode-only) and **the explorer role** (built-in `Explore`), so
the custom `explorer` agent was deleted rather than maintained alongside it. The same sweep
removed `test-agent` and the `runpod-ops` skill, both of which no command, skill, or fixture
referenced. A definition nothing invokes is not free: it still has to be kept true.

The check is mechanical. Grep the repository for each agent, skill, and command name,
excluding its own definition file. A name whose only remaining hit is the roster table in
`tooling-parity.md` is documentation describing itself, and is a candidate for removal.
