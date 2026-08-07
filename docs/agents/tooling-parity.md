# OpenCode ↔ Claude Code parity

This repo is usable from both OpenCode and Claude Code without content drift. Content,
policy, and prompt bodies are single-sourced; only the per-tool wiring is duplicated, and
each duplicated surface has a parity check.

Agent sessions themselves are sandboxed with no access to the host (see the execution
environment section in `AGENTS.md`): the parity, preflight, and sync `make` targets in this
document are run by the human on the workstation, and agents work from the reported output.

## Rules and style guides (Phase 1)

- Single source of truth: `AGENTS.md`. OpenCode reads it directly and ignores `CLAUDE.md`
  once `AGENTS.md` exists.
- `CLAUDE.md` is a thin importer: `@AGENTS.md` plus `@.styles/*.md` imports (Claude expands
  them at session start; OpenCode never reads the file).
- OpenCode loads the same style guides via `"instructions": [".styles/*.md"]` in
  `opencode.json`.
- Do NOT put `@` imports inside `AGENTS.md`: OpenCode does not parse `@`, so they would be
  dead text.

### Permission encoding (single policy, two files)

| Policy | Claude `.claude/settings.json` | OpenCode `opencode.json` |
|--------|-------------------------------|--------------------------|
| block `rm -rf *`, `nvidia-smi -r*` | `deny` Bash rules | `permission.bash: deny` |
| confirm `docker system prune`, `docker volume rm`, `git push`, model-dir `rm` | `ask` Bash rules | `permission.bash: ask` |

Known asymmetries — recorded so "the two wirings agree" stays honest:

- `Read(**/.env*)` and `Edit/Write(//mnt/ai-data/models/**)` **path**-denies exist only in
  the Claude encoding. OpenCode's permission schema is bash-pattern + coarse verbs and
  cannot express per-path globs; it covers these via the `* /mnt/ai-data/models*` bash
  rule and (Phase 4) the shared hook backstop.
- `share: "disabled"`, LSP config, and compaction settings are **OpenCode-only** (Claude
  Code has no equivalent keys and manages compaction automatically).
- Claude Code denies access outside the project directory by default; OpenCode makes the
  equivalent explicit with `external_directory: ask` in `opencode.json`.
- `doom_loop: deny` (kills the repeated-identical-call failure mode) is **OpenCode-only**;
  Claude Code has no equivalent key.

## Agents (Phase 2)

Ten roles, **authored once and generated**. `.claude/agents/<role>.md` is the source;
`.opencode/agents/<role>.md` is produced from it by `make agent-sync`, which copies the
prompt body verbatim and translates the frontmatter by the table below.

Do **not** symlink agent files: OpenCode passes unknown frontmatter keys through to the
provider as model options, so Claude-dialect keys would leak into live API calls. That is
one-directional, though — it rules out sharing one file, and it is precisely the argument
for a one-way projection, which is what the generator is.

`make agent-parity` no longer diffs the two trees after the fact. It regenerates and fails
if the result differs from what is committed, the same shape as `uv lock --check`: a
divergence is now prevented rather than detected. A role that genuinely needs to fork on the
OpenCode side may do so by deleting the generated-by marker from its frontmatter; the
generator then reports it as forked and leaves it alone, mirroring what `command-sync` does
for `cost.md`.

| Role | Purpose | Writes? |
|------|---------|---------|
| skeptic | Adversarial verification of one claim | no |
| bench-runner | Host-side service benchmarks, via human-reported output | no |
| doc-writer | Documentation prose | yes |
| bug-hunter | Defect discovery along one assigned dimension | memory only |
| triage-agent | Dedupe, severity, and packaging of a findings ledger | memory only |
| remediator | Implementing one verified finding, with red-before evidence | yes |
| integrator | Gate-running a merged batch and finding the seams | no |
| perf-auditor | Runtime and memory cost of repo-local code, in the sandbox | no |
| doc-auditor | Documentation drift detection | memory only |
| feature-implementer | Net-new work from a stated requirement, style-guide first | yes |

The six QA roles (`bug-hunter` through `doc-auditor`) share one method contract, the
`bug-bash` skill, and are driven by `/bugbash`. `skeptic` is the verification stage of
that same ladder. See `.claude/skills/bug-bash/SKILL.md`.

Two roles are deliberately kept unable to write. `integrator` is the independent check
between a batch and a commit, and `skeptic` adjudicates claims: an agent that can patch
what it finds stops being either. This is also why neither carries `memory` on the Claude
side — enabling it auto-grants Read, Write, and Edit.

Two roles were removed once the QA roster landed, both for the same reason: nothing
referenced them. `explorer` duplicated Claude Code's built-in `Explore`, which provides the
same context isolation (fan-out searching in a subagent, so raw file contents never enter
the primary's context) without a private definition to maintain. `test-agent` was cited
only by this table; its work now splits between `bug-hunter`, for whom a test gap is a
finding, and `remediator`, whose red-before evidence is a regression test. Prefer the
built-in `Explore` for search, and reach for a QA role for anything test-shaped.

### Frontmatter translation table

This table is executable: `scripts/agents/gen_opencode_agents.py` implements it, and an
unmapped key fails the generator rather than being silently dropped. Keep the two in step —
adding a row here means adding it there.

| OpenCode | Claude Code |
|----------|-------------|
| `description` | `description` (+ required `name`) |
| `mode: subagent` | implicit — everything in `.claude/agents/` is a subagent |
| `model: <provider>/<model>` | `model: opus \| inherit` |
| `temperature` | not supported — bake determinism into the prompt |
| `steps` | `maxTurns` |
| `permission` (per-tool allow/ask/deny) | `tools` allowlist / `disallowedTools` |
| `options.reasoningEffort` | `effort` |
| — | `memory`, `skills`, `color` (Claude-only) |

Two asymmetries in the QA roles, recorded so "the two wirings agree" stays honest:

- **`memory` is Claude-only.** `bug-hunter`, `triage-agent`, and `doc-auditor` keep a
  project-scoped memory directory (`.claude/agent-memory/<role>/`, version-controlled)
  so refuted claims and aspirational documents survive into the next campaign. OpenCode
  has no equivalent, so the memory steps in those bodies are inert there and the
  OpenCode permission is `edit: deny`. `feature-implementer` also declares `memory: project`
  but has accumulated no memory directory yet; the key is harmless until it writes one.

  **Every memory file carries an expiry.** Two frontmatter fields are mandatory:

  ```yaml
  verified: 2026-08-06        # when these claims were last checked against the tree
  stale_when: <observable event that invalidates this memory>
  ```

  `stale_when` names a condition a later agent can actually evaluate — "`ArchitectureSpec`
  gains per-layer-type fields", not "this gets fixed". Phase 2 of the bug-bash ladder
  re-grades memory against these fields before the finders run, and a memory whose trigger
  has fired is re-checked and rewritten rather than trusted.

  The reason is a failure this repository has already had. Memory is written in the
  imperative — "fold this into the known root rather than filing it as new" — so a memory
  that outlives its subject does not merely go quiet, it actively suppresses real findings.
  The 2026-08-06 audit found one memory instructing agents to treat a model config as
  contaminated months after the file had been re-downloaded and verified correct. Nothing
  had re-read it, because nothing was ever going to: memory is loaded by one role, and the
  facts it holds are invisible to everyone else. An expiry is what makes that survivable.
- **`effort: max` maps to `high`.** `bug-hunter` runs at `max` on the Claude side;
  OpenCode's `reasoningEffort` is set to `high`, the highest value the existing roles
  use.

The `skills` key preloads the `bug-bash` contract on the Claude side only, which is why
each QA body also names `.claude/skills/bug-bash/SKILL.md` by path — OpenCode discovers
the skill directory but does not preload from frontmatter.

### Model policy

**All roles pin the frontier model in both tools.** There is no low-cost or local tier: the
bench-runner and doc-writer subagents exist for **context isolation** (keeping benchmark
output and mechanical writing out of the primary's context), not to route work to a cheaper
model.

Claude side uses `model: opus`. OpenCode `model:` is a commented placeholder to be filled in
during manual OpenCode testing (`# model: <frontier provider/model>`), pointing at the same
frontier model.

Roles differ only in reasoning effort and step cap, never in model. The **skeptic** runs
`effort: high` + `maxTurns: 15` + OpenCode `temperature: 0` for deterministic judgment. The
QA roles run `effort: high` with step caps sized to their scope (25 to 40), except
**bug-hunter**, the one role at `effort: max`: discovery is the reasoning-bound stage, and a
defect a finder does not raise is one no verifier ever sees.

### Parity check

`make agent-parity` strips YAML frontmatter from each same-named agent in
`.claude/agents/` and `.opencode/agents/` and diffs the bodies; any mismatch is a non-zero
exit. It is wired into `make agent-preflight`, which Phase 6 extends with the
supply-chain scan (`scripts/agents/preflight-scan.sh`).

### Role acceptance tests

`tests/agents/` holds one fixture per role, each built so the wrong answer is the plausible
one: a broken "completed" task the skeptic must return `VERDICT: REFUTED` on, an off-by-one
behind a green test suite, a ledger entry with zero recorded verdicts, a speedup claim
smaller than its own noise floor, and a document that contradicts the tree. `make
agent-role-tests` runs them all through the case table in
`scripts/agents/role-acceptance.sh`; `ROLE=<name>` runs one. See `tests/agents/README.md`.

## Commands (Phase 3)

Slash-command template syntax is identical in both tools: `$ARGUMENTS`, positional
`$1..$N`, `` !`cmd` `` shell injection (backtick form), and `@path` file inclusion. Commands
are authored once in `.claude/commands/` and mirrored into `.opencode/commands/` by
**per-file symlinks** (`ln -s ../../.claude/commands/<name>.md`). `make command-sync`
(re)creates them; `make agent-parity` reports each `.opencode/commands/*.md` as `[OK]`
(symlink), `[INFO] forked` (real file), or `[ERROR]` (broken link).

Conventions for shared command files:

- **`allowed-tools: Bash(...)`** on any command using `` !`cmd` `` injection — Claude Code
  requires it for the injection to run; OpenCode ignores it.
- **`agent:`** routes the command to a subagent in OpenCode (e.g. `/audit-docs` →
  `doc-auditor`, `/bench` → `bench-runner`). Claude Code ignores it, so the body also says
  "delegate to the X subagent" and routing happens via delegation.
- **Omit `model:`** — the two tools use incompatible model identifiers. Pin models in the
  per-tool agent definitions instead.

Divergent commands are forked deliberately: break the symlink and author a real file.
`/cost` is the current example — OpenCode-only (`.opencode/commands/cost.md`, shell-injects
`` !`make agent-usage` ``) because Claude Code already ships a built-in `/usage`.

| Command | Shared? | OpenCode routing |
|---------|---------|------------------|
| `/task <roadmap> <id>` | shared | primary (implement → acceptance → skeptic pipeline) |
| `/audit-docs <path>` | shared | `agent: doc-auditor`, `subtask: true` |
| `/bench <service>` | shared | primary (owns the human handshake; delegates only the reduction to `bench-runner`) |
| `/verify-host <claim>` | shared | primary (a subagent has no channel to the human) |
| `/bugbash <scope>` | shared | primary (drives the phase ladder, spawns the QA roles) |
| `/postmortem [sprint]` | shared | primary (structured writing) |
| `/pins` | shared | primary (grep audit) |
| `/cost` | **forked, OpenCode-only** | primary (`make agent-usage`) |

## Skills (Phase 3)

Single location: **`.claude/skills/`**. Claude Code reads only this path; OpenCode discovers
it too (it also reads `.opencode/skills/`, which Claude ignores). Authoring here once serves
both tools with no duplication — this resolves the earlier open "decide one location"
question.

Frontmatter is kept to the shared subset — `name` (kebab, matching the directory) and
`description`. Skills are loaded on demand: only name + description sit in the available-skills
block until a skill is invoked.

Caveat: Claude-only keys such as `allowed-tools` are silently ignored by OpenCode, so **never
rely on skill frontmatter for safety** in OpenCode sessions — that is the job of the Phase 1
permissions and the Phase 4 hooks.

Current skills: `benchmarking`, `bug-bash`, `docker-debug`, `training-runs`.

A `runpod-ops` skill was removed: it described commands that do not exist and said so in
its own status note. The RunPod guardrail below is unaffected — it is a script plus two
hooks, and never depended on the skill.

## Guardrails and observability (Phase 4)

Guardrail *logic* is single-sourced as standalone scripts under `scripts/agents/`; each tool
is a thin adapter. The scripts read tool-call JSON on stdin and exit non-zero to block. This
is the belt-and-suspenders backstop over the Phase 1 permissions: permissions gate tools by
pattern, but the pattern matcher misses bash indirection (e.g. `cat $(echo .env)`) — the hook
catches it because it inspects the resolved command.

| Guardrail | Script | Claude Code | OpenCode |
|-----------|--------|-------------|----------|
| access guard | `scripts/agents/guard.sh` | `hooks.PreToolUse` (matcher `Bash\|Read\|Edit\|Write`), exit 2 blocks | `.opencode/plugin/guard.ts` on `tool.execute.before` |
| RunPod idle | `scripts/agents/runpod-idle-check.sh` | `hooks.Stop` + `hooks.SessionEnd` | `.opencode/plugin/runpod-idle.ts` on `session.idle` (toast) |
| token usage | — | built-in `/usage` | `.opencode/plugin/usage.ts` (writes usage state) |

- **guard.sh** blocks reads/writes of `*.env*`, writes under `/mnt/ai-data/models`, and any
  bash command that references an env file or writes under the models dir. Both adapters send
  the identical `{tool_name, tool_input}` shape, so the `.env`-via-indirection block is
  enforced the same way in both tools.
- **runpod-idle-check.sh** is non-blocking (always exit 0). It queries the RunPod GraphQL API
  with `RUNPOD_API_KEY`, `[SKIP]`s cleanly without a key, and supports `RUNPOD_IDLE_FIXTURE=
  <file.json>` to simulate a warm endpoint for testing. `RUNPOD_REPORT=1` emits a spend line
  for `make agent-usage`.
- **usage.ts** is OpenCode-only (Claude has the built-in `/usage`). It appends per-session
  token counts by provider/variant to the usage state file (default `.agent-state/usage.jsonl`,
  git-ignored; override `AGENT_USAGE_STATE`). A community plugin (`opencode-token-monitor`)
  can be wrapped instead of forked if richer accounting is wanted.

### One spend number: `make agent-usage`

Merges token usage (the OpenCode usage state, summed by provider/variant) with RunPod spend
(via `runpod-idle-check.sh` report mode) into a single report, so Phase 5's escalation-ladder
decisions read one number. It degrades cleanly when a source is absent.

Anthropic-side token usage is read in-session via the built-in `/usage`; it is intentionally
not folded into the merged report, as the only export path adds infrastructure not worth the
overhead on a single workstation.

### Notes

- OpenCode auto-loads local plugins from `.opencode/plugin/`; no `plugin` key is added to
  `opencode.json` for them (that key is for published packages). `.opencode/package.json` is
  kept near-dependency-free — bun installs plugins at startup, and startup time is inner-loop
  time. Confirm the plugin hook names against the installed OpenCode version on first run.
- Deferred (build only if manually cycling reasoning-effort proves tedious): a `chat.params`
  plugin that drops `reasoningEffort` for messages under a size threshold.

## Workflow automation (Phase 5)

The escalation ladder (batch → roadmap task → interactive) is documented once in
`docs/agents/README.md`; tiers map to reasoning-effort on the single frontier model. The
GPU-serialization + worktree-parallelism policy is a tool-agnostic operating rule in
`AGENTS.md` (both tools read it).

Batch parity is one make target with an `AGENT=opencode|claude` switch:
`make agent-run AGENT=... PROMPT=... [EFFORT=low]` maps to `opencode run --auto` ↔
`claude -p --output-format json --effort ...`. As of Phase 6 the `opencode` batch path
**refuses to run on the host** (its `--auto` auto-approves everything not denied) and points
at `make agent-sandbox`; only the Claude path runs on the host directly. The Claude path appends its JSON `usage` to the
usage ledger so `make agent-usage` shows the spend delta. Session export:
`make agent-export AGENT=... ID=...` (`opencode export` ↔ `claude -p --resume`), stored under
`logs/sessions/`. The `guard.sh` PreToolUse hook is the enforced safety floor in batch mode
regardless of permission flag; the exact permissive flag is version-dependent and overridable
via the `CLAUDE_PERM` make variable.

## Security hardening (Phase 6)

The Phase 1 permissions and the Phase 4 hooks are UX/backstop guardrails, not a security
boundary — OpenCode ships no Linux sandbox and has known bash-escape gaps, and both tools
auto-load repo-local supply-chain vectors with full user privileges. The boundary is
**OS-level containment**. Full threat model, tier split, and the quarterly review checklist
live in `docs/sandbox/agent-sandbox.md`; the parity-relevant wiring:

- **Containment (both tools, one image).** `docker/agents/Dockerfile.agents` pins both CLIs
  (`opencode-ai@1.17.18`, `@anthropic-ai/claude-code@2.1.207`); `docker-compose.agents.yaml`
  runs the session on an `internal: true` network whose only egress is an allowlisting Squid
  proxy (`docker/agents/proxy/`). `make agent-sandbox AGENT=claude|opencode NAME=<task>` mounts
  a git worktree read-write and injects exactly one scoped key. `--auto` and untrusted-input
  work run here; interactive work in this trusted repo may stay on the host.
- **Credential asymmetry (never single-sourced).** A Claude Code sandbox gets only
  `CLAUDE_CODE_OAUTH_TOKEN` (a subscription token minted by `claude setup-token`); an OpenCode
  sandbox gets only `RUNPOD_API_KEY`. Neither container gets
  the other's secret, and host credential dirs (`~/.ssh`, `~/.aws`, `~/.claude/`,
  `~/.local/share/opencode/auth.json`) are never mounted.
- **Permission tightening — the twin.** The webfetch/env-read narrowing is encoded in both
  files: `opencode.json` gets a `webfetch` allowlist (`docs.runpod.io`, `github.com`, else
  `ask`) plus bash denies (`cat *.env*`, `* ~/.ssh/*`, `* ~/.aws/*`);
  `.claude/settings.json` gets `Read(~/.ssh/**)` / `Read(~/.aws/**)` denies and
  `WebFetch(domain:docs.runpod.io|github.com)` allow with a broad `WebFetch` `ask`. (The
  invalid empty `lsp.python` key that blocked OpenCode from loading is fixed here too.)

| Concern | Script/config | Claude Code | OpenCode |
|---------|---------------|-------------|----------|
| supply-chain scan | `scripts/agents/preflight-scan.sh` | inspects `.claude/settings.json` hooks, `.claude/{skills,commands,agents}/` | inspects `opencode.json`, `.opencode/` (plugins/skills/commands) |
| containment | `docker-compose.agents.yaml` | `CLAUDE_CODE_OAUTH_TOKEN`, `DISABLE_AUTOUPDATER=1` | `RUNPOD_API_KEY`, `autoupdate: false` |
| server lockdown | — | no always-on localhost server (n/a) | `OPENCODE_SERVER_PASSWORD`, never `--hostname 0.0.0.0` |

The scan runs in two modes: `self` (the trusted home repo — advisory, exit 0) and `extern`
(any other checkout — every discovered vector is flagged and the scan exits non-zero). The
`extern` case is exercised by `tests/agents/preflight_fixture/`, which plants both an
`.opencode/plugin/evil.ts` and a `.claude/settings.json` hook; `make agent-preflight
PREFLIGHT_DIR=tests/agents/preflight_fixture` must flag both.
