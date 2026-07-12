# Agent sandboxing and security architecture

Guide for running coding agents (OpenCode and Claude Code) safely on the AMD Ryzen 9950X +
RTX 5090 workstation. It records the threat model, the containment design, the host-versus-
contained tier split, and a quarterly review checklist. It is the agent-side companion to the
inference-stack hardening in [README.md](README.md) and
[docker_compose_overview.md](docker_compose_overview.md).

## Table of contents

- [Conclusion first](#conclusion-first)
- [Threat model](#threat-model)
  - [Per-tool delta](#per-tool-delta)
- [Why the inference pattern does not copy directly](#why-the-inference-pattern-does-not-copy-directly)
- [Containment](#containment)
  - [Image and version pins](#image-and-version-pins)
  - [Egress allowlist](#egress-allowlist)
  - [Credentials and the never-mount list](#credentials-and-the-never-mount-list)
- [Tier split](#tier-split)
- [Configuration hardening](#configuration-hardening)
- [Supply-chain preflight](#supply-chain-preflight)
- [Quarterly review checklist](#quarterly-review-checklist)

## Conclusion first

The permission rules both CLIs enforce (Phase 1) and the shared guard hook (Phase 4) are
usability guardrails, not a security boundary. OpenCode ships no Linux sandbox and has known
bash-escape gaps in its permission enforcement, and both tools auto-load repo-local
configuration that runs with full user privileges. The boundary that holds is **OS-level
isolation**: a container with no host credentials, a writable git worktree instead of the real
tree, and an allowlisting egress proxy. Configuration hardening and hygiene sit on top of that
boundary, not in place of it.

## Threat model

- **Supply chain is the weakest link.** Opening an agent inside a cloned third-party repo can
  execute attacker-controlled code or inject attacker-controlled prompts. OpenCode auto-loads
  `.opencode/plugin/*` in-process with full privileges; Claude Code auto-loads
  `.claude/settings.json` hooks, skills, commands, and agents. Neither is signed or reviewed.
- **Prompt injection** arrives through webfetch and websearch output, MCP tool results, and
  untrusted repo files. Neither tool sanitizes that content before acting on it.
- **Local HTTP API history.** OpenCode's CVE record (an unauthenticated localhost HTTP server
  with permissive CORS allowing drive-by shell execution; a web-UI XSS escalating to command
  execution) shows the standing pattern: a local API bridged to a shell. Version currency and
  server lockdown are recurring requirements, not one-time fixes.
- **Exfiltration.** A compromised or injected session with open outbound network can send
  repository contents, credentials, or keys to an arbitrary host.

### Per-tool delta

Claude Code has a smaller always-on surface than OpenCode: it runs no persistent localhost
HTTP server, and it gates repo-local hooks behind a workspace trust prompt. That reduces the
drive-by-HTTP class of risk, but it does not change the supply-chain conclusion — a repo-local
`.claude/settings.json` hook, skill, or command is a vector morally identical to an
`.opencode/plugin/`. Both families are inspected together by the preflight scan.

Server lockdown items (`OPENCODE_SERVER_PASSWORD`, and never passing `--hostname 0.0.0.0`,
which exposes a server with no TLS) apply to OpenCode only. Claude Code has no equivalent
server to lock down.

## Why the inference pattern does not copy directly

The inference services are fixed-function appliances: load a model, serve an API, write
nothing but logs. Their hardening (`cap_drop: ALL`, `no-new-privileges`, `read_only`,
non-root, localhost-only `127.0.0.1:` binding) is correct and its philosophy transfers. A
coding agent is a general-purpose mutator, so four things differ:

1. **Writable git worktree instead of a `:ro` mount.** The agent's purpose is to edit code.
   Only a git worktree is mounted read-write (`../ai-experiments-<name>`), never `$HOME` or
   the repo root, so damage is `git`-recoverable.
2. **Egress control instead of inbound binding.** The appliances lock inbound traffic; an
   agent has no inbound port worth exposing, and its real risk is outbound exfiltration. This
   needs an egress allowlist, which the inference compose primitives cannot express.
3. **No GPU.** The NVIDIA container runtime widens the escape surface. The sandbox reserves no
   GPU; a benchmark task drives the existing GPU services over the network, honoring the
   one-GPU-consumer rule in `AGENTS.md`.
4. **Ephemeral run and one scoped secret.** The session is one-shot (`docker compose run
   --rm`), not a `restart: unless-stopped` daemon, and it carries exactly one scoped key.

## Containment

`docker-compose.agents.yaml` defines two services on two networks:

- `agent` runs on `agent-internal`, declared `internal: true`. The container therefore has no
  route to the internet except through the proxy. This network flag is the enforcement; the
  `HTTP_PROXY`/`HTTPS_PROXY` env in the image only tells well-behaved clients where the proxy
  is. The rootfs is `read_only: true`; the only writable surfaces are the bind-mounted
  worktree and tmpfs mounts for the home directory and caches.
- `agent-proxy` runs a Squid forward proxy on both `agent-internal` and `agent-egress` (the
  only network with real internet).

`make agent-sandbox AGENT=claude|opencode NAME=<task> [PROMPT="..."]` creates the worktree if
needed, injects the correct scoped key, and runs the session. Invoke it for every `--auto`
run, any webfetch-heavy or untrusted-input work, and any third-party checkout.

### Image and version pins

`docker/agents/Dockerfile.agents` installs both CLIs at exact versions:
`opencode-ai@1.17.18` and `@anthropic-ai/claude-code@2.1.207` (matching the host install at
authoring time), plus `pyright@1.1.411` (Python language server for both CLIs).
`DISABLE_AUTOUPDATER=1` and
the sandbox `opencode.json`'s `autoupdate: false` pair to freeze the pinned versions inside the
image — the image is the unit of currency, bumped by rebuild. On the host, by contrast, leave
updates on: given the CVE history, running stale is the larger risk.

`make agent-versions` compares each Dockerfile pin against the npm registry and reports which
pins are behind. It never edits the Dockerfile: bump one pin at a time and re-run the
acceptance checks, per the repository operating rules.

### Egress allowlist

`docker/agents/proxy/squid.conf` denies by default and allows only the domains an agent
legitimately needs: the Anthropic API and the Claude subscription-auth endpoints
(`claude.com`, `claude.ai` — used by `CLAUDE_CODE_OAUTH_TOKEN` sessions), the RunPod control plane (`api.runpod.io`,
`docs.runpod.io`), the code host (`github.com` and its asset domains), and the package
registries (npm, PyPI). HTTPS is allowlisted by CONNECT host without TLS interception, so no
private certificate authority is installed. The list grows only with a recorded reason.

### Credentials and the never-mount list

Each tool sees exactly one scoped secret, injected by environment:

| Tool | Injected key | Never present |
|------|--------------|---------------|
| Claude Code | `CLAUDE_CODE_OAUTH_TOKEN` (subscription token) | any RunPod key |
| OpenCode | `RUNPOD_API_KEY` (endpoint-scoped) | any Anthropic secret |

The Claude Code secret is a long-lived OAuth token minted once on the host with
`claude setup-token` (requires a claude.ai Pro/Max subscription; no Console API key). It is
scoped to inference and draws on the subscription quota, but has no per-token spend limit and
no server-side revocation today — store it in a password manager and export it only for the
sandbox invocation, for example `export CLAUDE_CODE_OAUTH_TOKEN=$(pass show anthropic/sandbox-token)`.
Never write it to the repository or to shell history.

Neither container's secret is the other's. The following are **never mounted** into a sandbox:
`~/.ssh`, `~/.aws`, `~/.claude/` (Claude Code credentials and user config), and
`~/.local/share/opencode/auth.json` (OpenCode auth). Verify the OpenCode auth store is
owner-only when it exists:

```bash
$ ls -l ~/.local/share/opencode/auth.json   # expect mode 600 (-rw-------)
$ chmod 600 ~/.local/share/opencode/auth.json
```

RunPod key hygiene: replace the single account-wide key with scoped keys — one endpoint-scoped
key that OpenCode sees, a separate read-only key for the `runpod-idle-check.sh` spend query,
and the account key stored nowhere on disk.

### Terminal fidelity and clipboard

The compose file passes `TERM` and `COLORTERM` through to the session container so the CLI
renders in truecolor; this is display fidelity only, with no security effect. The theme is
preseeded on every launch (`CLAUDE_THEME`, default `dark`, in the Makefile) because the
container home is an ephemeral tmpfs and `~/.claude.json` does not survive between sessions.

### Persistent settings

No Claude Code configuration survives between sessions: the container home is an
ephemeral tmpfs. Defaults therefore live in the Makefile, which seeds two files on every
launch: `~/.claude.json` (onboarding flag, theme, model options cache, and
`unpinFable5LaunchEffort` — without it the CLI pins a Fable session's launch effort to
its recommended value and silently overrides the seeded `effortLevel`) and
`~/.claude/settings.json` (persistent defaults — `model` from `CLAUDE_MODEL`, default
`claude-fable-5[1m]`; `effortLevel` from `EFFORT`, default `low`; and `theme` from
`CLAUDE_THEME`, default `dark`). Override per run:
`make agent-sandbox ... EFFORT=medium CLAUDE_MODEL=claude-opus-4-8`. In-session changes made with
`/effort` or `/model` last one session by design; to change a default permanently, change
the Makefile variable.

### Model selection

Account-gated models (Fable) appear in the `/model` picker only after an interactive OAuth
profile fetch populates `additionalModelOptionsCache` in `~/.claude.json`. The headless
`CLAUDE_CODE_OAUTH_TOKEN` auth path never performs that fetch, so the Makefile seeds the
cache entry (`CLAUDE_MODEL_OPTIONS`) into the per-session `~/.claude.json` alongside the
theme. A seeded entry is a picker affordance only: access is enforced server-side by the
token's subscription entitlement, so re-mint the token with `claude setup-token` if Fable
requests are rejected. `make agent-sandbox ... MODEL=<model-id>` passes `--model` directly
and works for both interactive and `PROMPT=` batch sessions.

The `/usage` panel inside a sandbox session shows only the Session block, without the
subscription quota bars. The bars require the interactive OAuth credential in
`~/.claude/.credentials.json`, which token-based auth never writes
(anthropics/claude-code#53735); check quota on the host or at claude.ai instead.

In-application copy shortcuts do not work in the sandbox, by design. Claude Code copies via
`xclip`/`wl-copy`/`pbcopy` (no OSC 52 support yet — anthropics/claude-code#20974), and the
container has neither a clipboard tool nor a display socket. Mounting an X11 or Wayland socket
would grant the contained agent read/write access to the host clipboard and input events, so
the socket stays out. Use terminal-native copy instead: select the text with the mouse and
press Ctrl+Shift+C — the terminal emulator runs on the host, so this path is unaffected.

## Tier split

| Work | Where |
|------|-------|
| Interactive session in this trusted repo | host, with the configuration hardening below |
| `--auto` / batch mode | contained (auto-approves everything not denied) |
| Untrusted input (webfetch-heavy research, third-party checkouts) | contained |
| Any future `opencode serve` usage | contained |

`make agent-run AGENT=opencode` refuses to run on the host and points at `make agent-sandbox`,
because its `--auto` mode inherits none of the ask-prompts that make host sessions tolerable.

## Configuration hardening

Best-effort layer on top of containment, encoded in both tools (see
[tooling-parity.md](../tooling-parity.md) for the twin):

- `opencode.json`: `webfetch` allowlist (`docs.runpod.io`, `github.com`, else `ask`); bash
  denies for `cat *.env*`, `* ~/.ssh/*`, `* ~/.aws/*` (the read-tool `.env` deny does not
  cover bash).
- `.claude/settings.json`: `Read(~/.ssh/**)` and `Read(~/.aws/**)` denies;
  `WebFetch(domain:docs.runpod.io)` and `WebFetch(domain:github.com)` allowed with a broad
  `WebFetch` set to `ask`.
- OpenCode server: set `OPENCODE_SERVER_PASSWORD` even for localhost use, and never pass
  `--hostname 0.0.0.0` (the server has no TLS; any remote exposure goes through an SSH tunnel,
  and preferably never leaves the sandbox).
- Plugin policy: exact-version pins for any published plugin; read the source of any community
  plugin before adopting it; prefer vendoring reviewed code into `.opencode/plugin/` over an
  npm reference. The Phase 4 plugins authored in this repo are trusted by construction.

## Supply-chain preflight

Before opening either tool in any repo not authored here — including submodule or upstream
updates — run:

```bash
$ make agent-preflight                                        # scans this repo (advisory)
$ make agent-preflight PREFLIGHT_DIR=/path/to/checkout        # scans an external tree
```

`scripts/agents/preflight-scan.sh` inspects both config families —
`opencode.json`, `.opencode/`, `AGENTS.md`, `.claude/settings.json`, `.claude/hooks*`,
`.claude/skills/`, `.claude/commands/`, `.claude/agents/`. In `self` mode (the trusted home
repo) it reports vectors for awareness and exits 0. In `extern` mode (any other tree) it flags
every discovered vector and exits non-zero, because external agent config is untrusted until
read. The `extern` path is covered by `tests/agents/preflight_fixture/`, which plants an
`.opencode/plugin/evil.ts` and a `.claude/settings.json` hook; the scan must flag both.

## Quarterly review checklist

- OpenCode and Claude Code security advisories published since the last check; bump the host
  installs and the image pins accordingly.
- Permission-gap status: has upstream Linux sandboxing for OpenCode landed? If so, evaluate
  whether it changes the containment requirement.
- Plugin pin bumps: review and update any vendored or referenced plugin versions.
- Key rotation: rotate the RunPod scoped keys and the Anthropic key; confirm the account-wide
  RunPod key still exists nowhere on disk.
- Re-run `make agent-preflight PREFLIGHT_DIR=tests/agents/preflight_fixture` and confirm it
  still flags both planted vectors.
