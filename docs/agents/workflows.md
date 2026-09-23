# Agent workflows

The recipes for running an agent: a host batch run, a contained session and the secrets it
needs, the shared docker footprint, the worktree lifecycle, session export, the
maintenance targets, and what to do when one of them refuses. Every target below prints
`[ERROR]` with a solution when a required variable or secret is missing.

## Table of contents

- [Host batch run](#host-batch-run)
- [Contained session](#contained-session)
- [Sandbox docker resources](#sandbox-docker-resources)
- [Worktree lifecycle](#worktree-lifecycle)
- [Session export](#session-export)
- [Maintenance targets](#maintenance-targets)
- [Troubleshooting](#troubleshooting)

## Host batch run

```bash
make agent-run PROMPT="Audit docker/ for unpinned image tags" EFFORT=low ALLOW='Read Grep'
```

Runs `claude -p` on the host at the given reasoning effort and appends token usage to
`.agent-state/usage.jsonl`. Only Claude Code runs on the host: `AGENT=opencode` refuses by
design, because OpenCode batch mode (`opencode run --auto`) auto-approves every action not
explicitly denied, and the refusal message redirects to the contained form below.

## Contained session

```bash
make agent-sandbox AGENT=claude NAME=mytask                  # interactive
make agent-sandbox AGENT=opencode NAME=mytask PROMPT="..."   # batch, contained
```

Creates the worktree `../ai-experiments-<NAME>` if it does not exist, then starts a
contained session (network-restricted container, egress proxy) in that worktree. Each tool
receives exactly one secret:

- `AGENT=claude` requires `CLAUDE_CODE_OAUTH_TOKEN`: generate once with
  `claude setup-token` on the host, then either export it or save it to the auto-loaded
  secrets file below.
- `AGENT=opencode` requires an endpoint-scoped `RUNPOD_API_KEY`.

To start every container authenticated without a per-shell export, save the token once to
a mode-600 file outside the repository that `make agent-sandbox` reads on the host.
Generate the token interactively first: `claude setup-token` opens a browser flow and
prints the token. Do not wrap it in `$(...)`; it is an interactive command, and a captured
value is the on-screen UI text, not a token.

```bash
claude setup-token            # complete the browser flow, then copy the printed token
mkdir -p ~/.config/ai-experiments
printf 'CLAUDE_CODE_OAUTH_TOKEN=%s\n' 'PASTE_TOKEN_HERE' \
    > ~/.config/ai-experiments/agent-secrets.env    # paste the token over the placeholder
chmod 600 ~/.config/ai-experiments/agent-secrets.env
```

The file must contain only unquoted `KEY=value` lines. An exported `CLAUDE_CODE_OAUTH_TOKEN`
still takes precedence; the file only fills in a secret not already in the environment, and
a malformed file is reported rather than executed. The same file may hold `RUNPOD_API_KEY`
for OpenCode sessions.

`CLAUDE_THEME` (default `dark`) and `EFFORT` (default `low`) preseed the Claude Code theme
and default reasoning effort each session, because the sandbox home is ephemeral tmpfs.
The persistent settings are described in
[../sandbox/agent-sandbox.md](../sandbox/agent-sandbox.md).

## Sandbox docker resources

The sandbox's docker footprint is shared across every worktree, not per-worktree. The make
wrapper pins the compose project to `ai-agents`, so every session, from any worktree, uses
one proxy container (`agent-proxy`), the `ai-agents_agent-internal` and
`ai-agents_agent-egress` networks, and two built images, `agent` and `agent-proxy`. Pinning
is required: the `agent-internal` subnet is fixed at `172.31.0.0/16`, so an unpinned
per-worktree project name collides with `Pool overlaps with other one on this address
space`. The per-session container is `--rm` and named `agent-<NAME>` while it runs, so
`docker ps` maps a live session to its worktree `../ai-experiments-<NAME>`.

`make agent-sandbox` runs `docker compose run` with no `--build`, so a change to the proxy
configuration, most often the egress allowlist in `docker/agents/proxy/squid.conf`, has no
effect until the image is rebuilt with `make agent-sandbox-build`. An allowlist edit that
appears to change nothing indicates a stale image rather than a wrong entry. A domain the
proxy does not carry surfaces as a transport error such as `Socket is closed`, not as a
permission message, so check the allowlist before diagnosing a network fault:
`docker exec agent-proxy cat /var/log/squid/access.log`.

Tear the shared footprint down with `make agent-sandbox-clean`. It runs
`docker compose -f docker-compose.agents.yaml -p ai-agents down --remove-orphans`, then
sweeps any leftover `*_agent-internal` and `*_agent-egress` networks from older, unpinned
runs; it never touches the inference stack. Add `--rmi local` to that compose command to
also drop the built images. If a crashed session leaves an `agent-<NAME>` container behind,
`make agent-sandbox-clean` removes it (or `docker rm agent-<NAME>`). `make clean` is scoped
to the inference stack, and the host-wide `docker system prune -f` lives in the explicit
`make prune` target.

## Worktree lifecycle

```bash
make worktree-mytask           # create ../ai-experiments-mytask on branch task/mytask
make worktree-mytask TIER=full # ... with the full environment (torch, training, GPU)
make worktree-ls               # list worktrees
make worktree-clean mytask     # remove when merged
```

A new worktree starts from the current `HEAD` commit. Uncommitted changes in the current
worktree do not reach it, so commit or stash before creating one. With `TIER=full` this
mistake stays invisible until after a multi-minute `flash-attn` build, and then presents
as a test suite failing against code that appears fixed.

Creating a worktree provisions its `.venv` automatically. `TIER` selects how much:

| `TIER` | Sync command | Use for |
|--------|--------------|---------|
| `sandbox` (default) | `uv sync --frozen --only-group dev --only-group sandbox` | Agent sessions and non-GPU work. Torch-free, provisions in seconds. |
| `full` | `uv sync --frozen --all-groups --all-extras` | The full test suite, training, and GPU work. Matches `make install`. |

The `sandbox` tier runs `make quality` and the non-GPU `make test` suite without a manual
sync. It does not install torch, so the torch-gated tests skip rather than fail;
`tests/conftest.py` prints `[TIER] sandbox:` or `[TIER] full:` beside the pytest summary,
and [../testing/test-suite.md](../testing/test-suite.md) says how to read it.

The `full` tier compiles CUDA extensions (`flash-attn` builds from an sdist against the
cu130 torch pin), so expect several GB of disk and a long build. `MAX_JOBS` and
`NVCC_THREADS` control build parallelism: `make worktree-mytask TIER=full MAX_JOBS=8`.

An unrecognised `TIER` fails with a styled `[ERROR]` before the worktree is created.
Provisioning itself is best-effort: if it cannot resolve (a stale `uv.lock`, no network, or
`uv` absent), the target prints a `[WARN]` with the manual `make install` command and
worktree creation still succeeds. A bare `uv sync` repairs a worktree environment safely:
`[tool.uv] default-groups` in `pyproject.toml` names `dev`, `sandbox`, `interp` and
`training`, so the reflexive command adds what is missing instead of removing the tiers it
was not told about. It installs torch into a `sandbox` tier worktree, which is the one
cost; `make install` is the fuller form, adding the `quant` extra.

Removal accepts the bare name, the `task/<name>` branch, a worktree path, `NAME=<name>`, or
the suffix form `make worktree-clean-<name>`; all resolve to the same worktree. A bare
`make worktree-clean` errors with usage instead of matching the creation pattern. Removal
reports each outcome separately: a worktree that is already gone prunes the stale
registration, reports `[SKIP]` and exits 0, so repeating the command is harmless; removal
from inside the target worktree is refused; a worktree holding uncommitted changes is
refused with those changes listed, and `FORCE=1` removes it anyway and discards them.

`make worktree-clean` removes only the git worktree; the docker resources are shared and
are torn down separately with `make agent-sandbox-clean`. Delete the task branch after
removal with `git branch -D task/mytask`; if it is kept, recreating the worktree reuses it.

Independent, non-GPU tasks parallelize cleanly as one session per worktree. What
serializes is governed by the operating rule in `AGENTS.md`: one GPU consumer at a time,
however many worktrees are open.

## Session export

```bash
make agent-export AGENT=claude ID=<session-id>
```

Writes the transcript to `logs/sessions/<id>.json`. The session ID is the CLI's
conversation identifier: for Claude Code, the UUID shown by `claude --resume` (the
transcript is also at `~/.claude/projects/<slug>/<id>.jsonl`); for OpenCode, the session ID
from `opencode sessions`. `/postmortem` cites the export.

| Action | OpenCode | Claude Code |
|--------|----------|-------------|
| Rung 1 batch | `opencode run --auto "<prompt>"`, contained only | `claude -p "<prompt>" --output-format json --effort low` |
| Rung 3 interactive | `opencode` (frontier, high effort) | `claude` (frontier, `--effort high`) |
| Session export | `opencode export <id>` | `claude -p --resume <id> --output-format json` |

## Maintenance targets

| Command | Purpose |
|---------|---------|
| `make agent-sync` | Regenerate `.opencode/agents/` from `.claude/agents/`; the OpenCode side is never edited by hand |
| `make agent-parity` | Fail when the generated OpenCode agents differ from what is committed, and report each `.opencode/commands` symlink as intact, forked or broken |
| `make command-sync` | Symlink `.claude/commands` into `.opencode/commands`; prunes links whose source was deleted |
| `make agent-preflight` | `command-sync`, `agent-parity`, and the supply-chain scan of both tool directories |
| `make agent-role-tests [ROLE=x]` | LLM acceptance test per subagent role; spends tokens |
| `make agent-skeptic-test` | The one role test that expects a `REFUTED` verdict |
| `make agent-usage` | The OpenCode token ledger and RunPod spend in one report; Claude Code usage is its built-in `/usage` and is not merged |
| `make agent-versions` | Compare the CLI pins in `docker/agents/Dockerfile.agents` against the npm registry, check-only |
| `make agent-sandbox-build` | Rebuild the `agent` and `agent-proxy` images after a Dockerfile, lockfile or allowlist change |

## Troubleshooting

- `[ERROR] CLAUDE_CODE_OAUTH_TOKEN not set`: run `claude setup-token` on the host, then
  either export the token or save it to `~/.config/ai-experiments/agent-secrets.env`
  (chmod 600) so `make agent-sandbox AGENT=claude` loads it automatically.
- Dangling symlinks in `.opencode/commands` (source command deleted): run
  `make command-sync` to prune them; `make agent-parity` reports them as broken.
- `make agent-parity` fails after editing a role: run `make agent-sync` and commit the
  regenerated OpenCode file with the change.
- A fetch fails with `Socket is closed` inside the sandbox: the domain is not on the
  egress allowlist, or the proxy image is stale; see the docker resources section above.
- `make test` exit code 5: pytest collected zero tests, a pytest convention rather than a
  Makefile failure. The agent fixtures it runs are described in
  `tests/agents/README.md`.

## Related

- [README.md](README.md) — the ladder, and the generated roster of roles, commands and skills
- [roles.md](roles.md) — what each role is for and why some cannot write
- [tooling-parity.md](tooling-parity.md) — how the two tools share one set of definitions
