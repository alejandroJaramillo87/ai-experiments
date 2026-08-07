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

From the agent's side the containment is absolute: a sandboxed session has no route to the
host, its Docker daemon, its GPU, or the running services. All host-side validation —
`make agent-preflight`, `make agent-parity`, service health, benchmarks — is executed by the
human on the workstation, who reports the output back to the agent as the evidence of record.

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
`opencode-ai@1.17.18` and `@anthropic-ai/claude-code@2.1.212` (matching the host install at
authoring time), plus `pyright@1.1.411` (Python language server for both CLIs).
`DISABLE_AUTOUPDATER=1` and
the sandbox `opencode.json`'s `autoupdate: false` pair to freeze the pinned versions inside the
image — the image is the unit of currency, bumped by rebuild. On the host, by contrast, leave
updates on: given the CVE history, running stale is the larger risk.

The proxy is pinned on the same principle, and more strictly, because it is the component that
enforces the allowlist: `docker/agents/proxy/Dockerfile.proxy` fixes both the Debian base
(`bookworm-20250203-slim`) and `squid` itself to an exact package version. That pin comes from
`bookworm-security`, which supersedes rather than accumulates, so a later security release
removes the pinned one from the archive and the build fails by name — the intended prompt to
read the advisory and bump.

`make agent-versions` compares each Dockerfile pin against the npm registry and reports which
pins are behind. It also checks that `.opencode/package.json` pins `@opencode-ai/plugin` to the
same version as `OPENCODE_VERSION`: that package publishes in lockstep with the CLI and the
plugins under `.opencode/plugin/` are typed against the runtime that loads them, so bumping one
without the other is the drift worth catching. The script never edits anything: bump one pin at
a time and re-run the acceptance checks, per the repository operating rules.

### Python toolchain

The image carries `uv 0.11.28` (from `ghcr.io/astral-sh/uv`, matching the workstation), a
managed CPython `3.12.11` satisfying `pyproject.toml`'s `requires-python`, and an environment
at `/opt/pyenv` built by `uv sync --frozen --only-group dev --only-group sandbox` against
`uv.lock`. The dev group supplies `ruff`, `mypy` with `types-pyyaml`, `pytest`, and
`shellcheck` (the `shellcheck-py` wheel, not an apt copy); the sandbox group supplies the
GPU-less test extent. Because the versions are resolved from the lockfile rather than
transcribed, they are the workstation's by construction. An agent can therefore lint and test
its own edits instead of routing every Python change through the workstation for its first
check.

The runtime rootfs is read-only and the home directory is a 256MB tmpfs, so the uv paths are
fixed at build time: interpreters and tool environments under `/opt/uv`, executables in
`/usr/local/bin`, and the cache in `/tmp`. A cache written to the default `$HOME` location
would not survive the tmpfs.

`make quality` and `make test` are therefore the same commands here as on the workstation:
`make/quality.mk` branches on `AI_SANDBOX`, which this image sets, and drops the
`uv run --no-sync` prefix so every tool comes from `/opt/pyenv`. The compose file masks the
worktree's `.venv` with a tmpfs, so the workstation's environment can be neither read nor
written from inside. One gate step stays out of reach: `uv lock --check` resolves against the
pytorch cu130 index, which the allowlist does not carry, and the recipe prints an explicit
`[SKIP]` line for it. Read that line rather than the exit code; the workstation gate covers it
before every push.

### Rust toolchain

The image carries the toolchain pinned by `rust-harness/rust-toolchain.toml` (currently
`1.88.0`, matching the workstation), copied from the published `rust:1.88.0-slim-bookworm`
image rather than fetched by rustup at build time, plus `clippy`, `rustfmt`, `make`, and the
`gcc`/`libc6-dev` a linker needs. The workspace's dependency graph is resolved at image build
with `cargo fetch --locked`, so `make -C rust-harness quality` runs offline.

`CARGO_HOME` cannot live on the read-only rootfs or in the 256MB home tmpfs, so it is `/tmp/cargo`,
seeded once per session from the baked registry by `/etc/profile.d/cargo-home.sh`; the make
wrapper launches a login shell, which sources it. Build artifacts are not in tmpfs — `target/`
defaults into the bind-mounted worktree.

The gate also needs `numpy` and `gguf`, at the `pyproject.toml` pins, in their own venv first
on `PATH`. This is not optional: `runtime/src/npz.rs` shells out to `python3` to re-read a
written shard with `np.load`, so `cargo test` fails outright without numpy rather than
skipping. Read the gate's output for `[SKIP]` lines rather than trusting its exit code, and
note that Tier 3 tests still require the workstation.

### Staging model weights

The sandbox mounts only the worktree, so a checkpoint has to live inside it to be reachable.
Copy one under `models/` in the worktree; `.gitignore` covers `models/`, `*.gguf`, and
`*.safetensors`, with `rust-harness/fixtures/*.gguf` negated so the committed test fixtures
stay tracked. This deliberately avoids mounting `/mnt/ai-data`, which stays on the never-mount
list.

Two consequences follow from the copy. Each worktree needs its own, unless the worktree and the
model store share a filesystem and the file is hardlinked instead. And the CPU backend
dequantizes every tensor to f32 at load, so resident cost far exceeds on-disk size: E2B's body
plus main embedding is roughly 2.28B parameters (~9.1GB as f32) and E4B's roughly 4.67B
(~18.7GB). The container limit is 32GB for that reason.

Header inspection needs none of that memory. `gguf-info` reads only the metadata and tensor
directory, never the weight payload, so architecture reconciliation and tokenizer questions are
answerable well inside the limit.

### Egress allowlist

`docker/agents/proxy/squid.conf` denies by default and allows only the domains an agent
legitimately needs: the Anthropic API and the Claude subscription-auth endpoints
(`claude.com`, `claude.ai` — used by `CLAUDE_CODE_OAUTH_TOKEN` sessions), the RunPod control plane (`api.runpod.io`,
`docs.runpod.io`), the code host (`github.com` and its asset domains), and the package
registries (npm, PyPI). It also carries the reference sources a session needs to avoid guessing:
Hugging Face (`huggingface.co`) for model configurations, tokenizers, and safetensors headers;
`crates.io` and `docs.rs` for Rust dependency metadata and its rendered documentation; and
`ai.google.dev` for Google model documentation.
HTTPS is allowlisted by CONNECT host without TLS interception, so no private certificate
authority is installed. The list grows only with a recorded reason.

Every entry uses the dotted form (`.huggingface.co`), which matches the apex domain and all
subdomains; squid rejects an entry that is already a subdomain of another, so a domain and its
subdomain cannot both appear. `ai.google.dev` is the exception, listed without a leading dot so
that one documentation site does not open every `google.dev` subdomain.

Hugging Face needs two entries, not one. Configuration and tokenizer JSON are served from
`huggingface.co` directly, but a file read under `/resolve/` returns a 302 to Xet-backed CDN
storage on `*.cdn.hf.co`, so reading a safetensors header by range request crosses two domains.
With only the base domain allowed, the first hop succeeds and the redirect hop returns
`403 Forbidden` with `X-Squid-Error: ERR_ACCESS_DENIED` — which reads as a Hugging Face failure
rather than a local policy refusal. Check the response headers when diagnosing this: the proxy
identifies itself as `Server: squid/5.7`.

The redirect target is regional (`us.aws.cdn.hf.co`), which is why the entry is the dotted form
rather than an exact host.

**A change to `squid.conf` takes effect only after the proxy image is rebuilt**, which
`make agent-sandbox-build` does — it builds both `agent` and `agent-proxy`. `make agent-sandbox`
itself runs `docker compose run` with no `--build`, so an edited allowlist that appears to have
no effect means the build step was skipped.

The symptom of a domain that the allowlist does not carry is a transport error such as
`Socket is closed`, not a permission message: squid cannot place an explanation inside a CONNECT
tunnel it refused to open. That is unfixable at the proxy, so the list is instead stated up front
in the sandbox banner — `make/agents.mk` derives it from this file with `sed` and seeds it into
`~/.claude/CLAUDE.md`, so a session knows the boundary before it tries to cross it. When a fetch
still fails that way, confirm with
`docker exec agent-proxy cat /var/log/squid/access.log`.

### Credentials and the never-mount list

Each tool sees exactly one scoped secret, injected by environment:

| Tool | Injected key | Never present |
|------|--------------|---------------|
| Claude Code | `CLAUDE_CODE_OAUTH_TOKEN` (subscription token) | any RunPod key |
| OpenCode | `RUNPOD_API_KEY` (endpoint-scoped) | any Anthropic secret |

The Claude Code secret is a long-lived OAuth token minted once on the host with
`claude setup-token` (requires a claude.ai Pro/Max subscription; no Console API key). It is
scoped to inference and draws on the subscription quota, but has no per-token spend limit and
no server-side revocation today. Store it in one of two places — never in the repository or in
shell history:

- A password manager, exported only for the sandbox invocation, for example
  `export CLAUDE_CODE_OAUTH_TOKEN=$(pass show anthropic/sandbox-token)`.
- A mode-600 file outside the repository, `~/.config/ai-experiments/agent-secrets.env`, which
  `make agent-sandbox` reads on the host when the token is not already in the environment (an
  exported value still takes precedence). Generate the token interactively with
  `claude setup-token` and paste it in — do not capture it with `$(...)`, which is an interactive
  flow and would store on-screen UI text, not a token — then write a single unquoted line
  `CLAUDE_CODE_OAUTH_TOKEN=<token>`. The wrapper extracts only that key (it never executes the
  file, and reports a malformed value instead of failing obscurely), so this removes the
  per-shell export step and every container starts authenticated. The file is read on the host
  and only the token value crosses into the container as an environment variable; the file
  itself is never mounted. The same file may hold `RUNPOD_API_KEY` for OpenCode sessions.

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
ephemeral tmpfs. Defaults therefore live in the Makefile, which seeds three files on every
launch: `~/.claude.json` (onboarding flag, theme, model options cache, and
`unpinFable5LaunchEffort` — without it the CLI pins a Fable session's launch effort to
its recommended value and silently overrides the seeded `effortLevel`);
`~/.claude/settings.json` (persistent defaults — `model` from `CLAUDE_MODEL`, default
`claude-opus-5[1m]`; `effortLevel` from `EFFORT`, default `low`; and `theme` from
`CLAUDE_THEME`, default `dark`); and `~/.claude/CLAUDE.md` (a sandbox banner loaded as
user-scope memory, stating up front that the session is contained — no host, GPU, Docker
daemon, `/mnt/ai-data`, or running services — so the agent does not spend a turn
rediscovering its limits). Because the banner is seeded into the tmpfs home, it appears only
inside the container and never touches the host's own `~/.claude/CLAUDE.md`. Override per run:
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
[tooling-parity.md](../agents/tooling-parity.md) for the twin):

- `opencode.json`: `webfetch` allowlist (`docs.runpod.io`, `github.com`, else `ask`); bash
  denies for `cat *.env*`, `* ~/.ssh/*`, `* ~/.aws/*` (the read-tool `.env` deny does not
  cover bash).
- `.claude/settings.json`: `Read(~/.ssh/**)` and `Read(~/.aws/**)` denies; a per-domain
  `WebFetch` allow list with a broad `WebFetch` set to `ask`; denies for the host-only tools
  (`docker`, `nvidia-smi`, `make up|down|status|logs|shell-*`) so that a command which cannot
  work from a contained session fails as policy rather than obscurely; and allows for the
  pinned repo-local tools (`ruff`, `mypy`, `shellcheck`, `pytest`, `cargo`), which the gate
  invokes without the `uv run --no-sync` prefix here. The `WebFetch` domain list is kept in agreement with
  `docker/agents/proxy/squid.conf`: a domain permitted in one and absent from the other
  produces a confusing transport error instead of a clear refusal.
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
- Allowlist drift: diff the three lists of permitted domains — `squid.conf`,
  `.claude/settings.json`, and `opencode.json`. They serve different purposes (squid enforces,
  the other two are convenience filters), so they need not be identical, but a domain permitted
  by a tool and absent from squid produces a transport error rather than a refusal. There is no
  automated check by design: three lists reviewed quarterly does not justify one.
- Proxy pins: confirm `squid` and the Debian base tag in `docker/agents/proxy/Dockerfile.proxy`
  are still the intended versions. The proxy is the sole arbiter of egress.
- Key rotation: rotate the RunPod scoped keys and the Anthropic key; confirm the account-wide
  RunPod key still exists nowhere on disk.
- Re-run `make agent-preflight PREFLIGHT_DIR=tests/agents/preflight_fixture` and confirm it
  still flags both planted vectors.
