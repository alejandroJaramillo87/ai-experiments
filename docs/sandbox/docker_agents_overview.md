# Agent sandbox container implementation

Explanation of the contained agent session as its files build it: the agent image
(`docker/agents/Dockerfile.agents`), the egress proxy (`docker/agents/proxy/Dockerfile.proxy`
and `squid.conf`), and the two services, two networks, and limits in
`docker-compose.agents.yaml`. The threat model and the reasoning behind each control are in
[agent-sandbox.md](agent-sandbox.md); this page walks the implementation, line by line, the
way the other pages here walk an inference service.

> Note: This document explains the actual implementation in `docker/agents/Dockerfile.agents`,
> `docker/agents/proxy/Dockerfile.proxy`, `docker/agents/proxy/squid.conf`,
> `docker/agents/opencode.sandbox.json`, and `docker-compose.agents.yaml`, invoked by
> `make agent-sandbox` in `make/agents.mk`. Those files are the source of truth; this document
> follows them.

## Table of Contents

<!-- toc:start -->

1. [Design](#design)
2. [The agent image](#the-agent-image)
   1. [Toolchain sources](#toolchain-sources)
   2. [CLI pins](#cli-pins)
   3. [Runtime packages](#runtime-packages)
   4. [Python toolchain](#python-toolchain)
   5. [Rust toolchain at run time](#rust-toolchain-at-run-time)
   6. [Login-shell PATH](#login-shell-path)
   7. [Environment and user](#environment-and-user)
3. [The proxy image](#the-proxy-image)
4. [Compose integration](#compose-integration)
   1. [The proxy service](#the-proxy-service)
   2. [The agent service](#the-agent-service)
   3. [Networks](#networks)
5. [The wrapper](#the-wrapper)
6. [Interface](#interface)
7. [Reference implementation](#reference-implementation)

<!-- toc:end -->

## Design

An agent is a general-purpose mutator: it edits a repository, runs compilers, and installs
packages. The security model is therefore OS-level isolation, not the permission rules the
CLIs enforce internally. Three facts carry it:

- The `agent` service sits on an `internal: true` network with no route to the internet
  except the proxy, and the proxy forwards only to an allowlist of domains.
- The rootfs is read-only; only a bind-mounted git worktree and tmpfs caches are writable,
  so a compromised session's damage is git-recoverable.
- No GPU, no host credential mounts; exactly one scoped key is injected per session.

| Service | Container | Image | Networks |
|---------|-----------|-------|----------|
| `agent-proxy` | `agent-proxy` | `Dockerfile.proxy` | `agent-internal`, `agent-egress` |
| `agent` | `agent-<name>`, one per `run --rm` | `Dockerfile.agents` | `agent-internal` |

The compose file is separate so `make up` never starts it. `make agent-sandbox` is the
wrapper; a bare `docker compose -f docker-compose.agents.yaml run --rm agent` works but
without the worktree, secret, and seeded-settings handling the wrapper supplies.

## The agent image

### Toolchain sources

Three `FROM` stages, each pinned by tag:

```dockerfile
ARG UV_VERSION=0.11.28
ARG RUST_VERSION=1.88.0
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-source
FROM rust:${RUST_VERSION}-slim-bookworm AS rust-source
FROM node:22.14.0-bookworm-slim
```

Both `ARG`s precede the first `FROM` because a build argument used in a `FROM` must be
declared in the global scope, or it expands to empty and the stage name is invalid. uv and
Rust come from published images rather than curl installers, so each is pinned by a
digest-resolvable tag like every other image in the repository. The Rust tag is
`-slim-bookworm` so the copied binaries link against the same Debian 12 glibc 2.36 the node
base ships. 1.88.0 matches the workstation and `rust-harness/rust-toolchain.toml`, so a
sandbox gate result means what a workstation result means; the workspace's declared minimum
is 1.87, and the pin is deliberately the host's version rather than that floor.

The `rust-source` stage adds `clippy` and `rustfmt`, which the minimal rustup profile does
not ship and `make quality` runs, and prefetches the workspace's crate registry with
`cargo fetch --locked` so the gate stays offline at run time. Only the manifests are copied,
`Cargo.toml` and `Cargo.lock` for the workspace and each member, so this layer rebuilds when
the lockfile changes and not when source does. cargo refuses to parse a manifest whose target
file is missing, so the stage writes stub `main.rs` and `lib.rs` files, fetches, and deletes
the extracted sources under `/opt/cargo/registry/src`; the stubs are discarded with the stage.

### CLI pins

```dockerfile
ARG OPENCODE_VERSION=1.17.18
ARG CLAUDE_CODE_VERSION=2.1.212
ARG PYRIGHT_VERSION=1.1.411
ARG PYTHON_VERSION=3.12.11
```

Both CLIs and the pyright language server install globally with `npm install -g` at exact
versions, then `npm cache clean --force`. `make agent-versions` compares these pins against
the npm registry. Version currency is managed by rebuilding against a new pin, never by
in-place self-update: `DISABLE_AUTOUPDATER=1` covers Claude Code, and the OpenCode
configuration copied to `/etc/opencode/opencode.json` and selected through `OPENCODE_CONFIG`
sets `autoupdate` false and `share` disabled. That file lives on the read-only rootfs so the
agent's home directory can be an ephemeral tmpfs.

### Runtime packages

`git` for the worktree workflow; `curl` and `ca-certificates` for egress and the acceptance
checks; `jq`, `ripgrep`, and `make`; `gcc` and `libc6-dev` for the few source distributions
in the dependency groups. `shellcheck` is deliberately absent: the `dev` group pins
`shellcheck-py`, which is the binary the workstation gate runs, and an apt copy on `PATH`
would be a second version reachable by the same command.

### Python toolchain

uv installs the interpreter under `/opt/uv/python` and creates `/opt/pyenv` from the
lockfile with `uv sync --active --frozen --only-group dev --only-group sandbox`, the same form
`make/agents.mk` uses to provision a `TIER=sandbox` worktree, so the image and that tier are
the same environment by construction. `--only-group` rather than `--group`: the latter adds
the groups to the project dependencies and would drag in torch and the CUDA stack, gigabytes
the image has no use for. `sandbox` is the GPU-less test extent and `dev` carries ruff, mypy,
shellcheck, and pytest. Two members are load-bearing beyond testing: `rust-harness/runtime/src/npz.rs`
shells out to `python3` to re-read a written shard with `np.load`, so `cargo test` fails
rather than skips without numpy, and the cross-language contract tests need `gguf` to
exercise the real Python exporters.

Every path uv writes is pinned because the rootfs is read-only and `$HOME` is a small tmpfs
at run time: `UV_PYTHON_INSTALL_DIR` and `UV_TOOL_DIR` under `/opt`, baked into the image;
`UV_CACHE_DIR=/tmp/uv-cache`, on the writable tmpfs. The manifests are kept under
`/opt/sandbox-env` as the image's record of what `/opt/pyenv` was built from. The image
therefore depends on `uv.lock`, and `make agent-sandbox-build` is required after a dependency
change, as it already is after a `Cargo.lock` change.

### Rust toolchain at run time

`/usr/local/rustup`, `/usr/local/cargo`, and the prefetched registry at `/opt/cargo` are
copied from the builder stage. `CARGO_HOME` must be writable and can live neither on the
rootfs nor in `$HOME`, so it is `/tmp/cargo`; `/etc/profile.d/cargo-home.sh` seeds it from
`/opt/cargo` once per session. A session that skips the seed still works, because cargo falls
back to crates.io, which the proxy allowlists; it is merely slower.

### Login-shell PATH

The make wrapper launches `sh -lc`, and Debian's `/etc/profile` assigns `PATH`
unconditionally, discarding the image's `ENV PATH` additions. That once left a session with no
`cargo` at all and a `python3` with no numpy, while ruff, mypy, and shellcheck survived only
by sitting in the default `PATH`, which made the breakage look like a missing toolchain.
`/etc/profile.d/10-agent-path.sh` prepends `/opt/pyenv/bin` and `/usr/local/cargo/bin` after
that assignment. Both mechanisms are kept: `ENV` for non-login invocations, the profile
fragment for login shells.

### Environment and user

| Variable | Value | Purpose |
|----------|-------|---------|
| `AI_SANDBOX` | `1` | Marks a contained session; `make/quality.mk` reads it to invoke the pinned tools bare and to announce the lock check it cannot run |
| `ENABLE_LSP_TOOL` | `1` | Claude Code's native LSP tool; pyright serves both CLIs |
| `DISABLE_AUTOUPDATER` | `1` | See CLI pins |
| `HTTP_PROXY`, `HTTPS_PROXY`, lower-case forms | `http://agent-proxy:3128` | Tells well-behaved clients where the proxy is; defense in depth only, the network is the enforcement |
| `NO_PROXY`, `no_proxy` | `localhost,127.0.0.1,agent-proxy` | |
| `OPENCODE_CONFIG` | `/etc/opencode/opencode.json` | The sandbox-only OpenCode configuration |
| `CARGO_HOME` | `/tmp/cargo` | See above |

`USER node`, uid 1000, is the user the node base already ships and matches the host user that
owns the bind-mounted worktree, so writes to it need no root. `WORKDIR /work`. There is no
`ENTRYPOINT`; `CMD ["bash"]` is overridden by the wrapper, so one image serves both tools.

## The proxy image

`Dockerfile.proxy` builds on `debian:bookworm-20250203-slim` and installs
`squid=5.7-2+deb12u6` exactly, because this container is the sole arbiter of egress for every
session: an unversioned install would let the component enforcing the allowlist change under
a pinned base. The version is what bookworm-security offers, and that archive supersedes
rather than accumulates, so when a later revision lands the build fails with "Version not
found", which is the loud, correct signal to review the advisory and bump. `ca-certificates`
is left unpinned as a data package whose purpose is to carry the current trust store. The
image `EXPOSE`s 3128 and runs `squid -N -d 1`: foreground as PID 1, logging to stderr.

`squid.conf` is a deny-by-default forward proxy. Only the `172.31.0.0/16` subnet may use it;
CONNECT is refused to anything but 443 and any request to a port other than 80 or 443 is
denied. HTTPS is allowlisted by CONNECT host with no TLS interception, so no private CA is
installed and no traffic is decrypted. A leading dot on a `dstdomain` entry covers the apex
and every subdomain, and only the dotted form is listed because squid rejects an entry that
is already a subdomain of another. The proxy caches nothing, deletes the `X-Forwarded-For`
header and the `Via` header, and writes an access log of every reached domain to
`/var/log/squid/access.log`, readable with `docker exec agent-proxy cat` on that path.
[agent-sandbox.md](agent-sandbox.md) records why each domain is on the list.

## Compose integration

### The proxy service

`agent-proxy` runs with `no-new-privileges`, `cap_drop: ALL`, and `read_only: true`, adding
back only `SETUID` and `SETGID`, which squid needs to bind its port and drop to its helper
user. It is bounded because it is the container with real internet, and an unbounded proxy
defeats the point of bounding the thing it proxies for:

| Limit or mount | Value |
|----------------|-------|
| `memory` | 256M |
| `pids` | 256 |
| `/var/spool/squid` | tmpfs, 64M |
| `/var/log/squid` | tmpfs, 32M |
| `/var/run` | tmpfs, 16M |

It joins both networks: real internet through `agent-egress`, reachability from the sandbox
through `agent-internal`. `restart: "no"`; `make agent-sandbox-clean` tears it down.

### The agent service

`agent` runs with the same three hardening settings and no added capabilities. The writable
surfaces are the bind-mounted worktree and three tmpfs mounts, and nothing else. The tmpfs
flags `noexec`, `nosuid`, and `nodev` are Docker's defaults, spelled out because they are
load-bearing: `tests/conftest.py` relocates pytest's temporary directory after probing for
them, and `AGENTS.md` tells agents to build under `/work` because nothing else can execute. A
switch to the long-form mount syntax would drop them silently.

| Mount | Size | Purpose |
|-------|------|---------|
| `/home/node` | 256M, mode 0700, uid and gid 1000 | Ephemeral home; the seeded CLI settings live here |
| `/tmp` | 4G, mode 1777 | Caches, and `CARGO_HOME` seeded from the prefetched registry; 1G left no headroom once the dependency tree unpacked |
| `/work/.venv` | 16M, mode 0755 | Masks the worktree's `.venv`, which points at a host interpreter that does not exist here; small on purpose, so an actual install exhausts it and says so rather than building a second environment over the mount |
| `${AGENT_WORKTREE}` | bind, read-write | `/work`, the git worktree; damage stays inside it |

`AGENT_WORKTREE` is required with no default (`${AGENT_WORKTREE:?...}`): a bare
`docker compose run agent` would otherwise mount the current directory, the repository root
with `.env` included. The build and teardown targets pass a placeholder, since neither uses
the mount.

The resource limits all sit under `deploy.resources.limits`, because compose refuses to see
both the short `cpus:` form and the `deploy` form in one service, and memory was already
there:

| Limit | Value | Reason |
|-------|-------|--------|
| `memory` | 32G | 8G covers editing, linting, and the rust gate against the tiny fixture; the CPU backend dequantizes every tensor to f32 at load, so E2B costs roughly 9.1 GB and E4B roughly 18.7 GB before the KV cache. 32G fits E4B with headroom and leaves the 128 GB host three quarters free |
| `cpus` | 24.0 | A bug ceiling, not an attack ceiling: a runaway `cargo build -j` degrades the session, not the workstation, and an unattended session is when that happens. 24 of the 9950X's 32 threads keeps the rust gate parallel and the host responsive |
| `pids` | 4096 | The fork-bomb ceiling, on the same reasoning |

Environment: exactly one scoped secret is populated by the wrapper and the other is left
empty. A Claude Code session carries `CLAUDE_CODE_OAUTH_TOKEN`, a subscription token minted
by `claude setup-token` on the host, with no API key and no `~/.claude` mount; an OpenCode
session carries `RUNPOD_API_KEY`. `TERM` and `COLORTERM` pass through for rendering only.
`depends_on: agent-proxy`, `working_dir: /work`, `restart: "no"`, and a `command` of `bash`
that the wrapper overrides with the CLI invocation.

### Networks

| Network | `internal` | Members | Purpose |
|---------|------------|---------|---------|
| `agent-internal` | true | `agent`, `agent-proxy` | No internet; containers reach only each other. Fixed subnet `172.31.0.0/16` so the proxy's `localnet` ACL matches |
| `agent-egress` | false | `agent-proxy` | Real egress, used only by the proxy |

`internal: true` on `agent-internal` is the enforcement; the `HTTP_PROXY` variables in the
image are convenience. Because the subnet is fixed, only one compose project may hold the
network at a time, so the wrapper pins the project name to `ai-agents` (`AGENTS_PROJECT`)
rather than letting it default to the per-worktree directory name, which would fail across
worktrees with "Pool overlaps with other one on this address space".

## The wrapper

`make agent-sandbox AGENT=claude|opencode NAME=<task>` creates the worktree
`../ai-experiments-<name>` if absent, loads the one secret the chosen agent needs from the
environment or from `~/.config/ai-experiments/agent-secrets.env`, and runs
`docker compose -f docker-compose.agents.yaml -p ai-agents run --rm --name agent-<name> agent sh -lc "<cmd>"`
with `AGENT_WORKTREE` set to the worktree and the other agent's secret set to empty. For a
Claude Code session the command first seeds `~/.claude.json`, `~/.claude/settings.json`
(model from `CLAUDE_MODEL`, effort from `EFFORT`, theme), and a `~/.claude/CLAUDE.md` banner
naming what the sandbox has, what it lacks, and the exact egress allowlist read from
`squid.conf`, so a session does not spend a turn rediscovering its limits. The home is tmpfs,
so nothing of this survives the session.

## Interface

| Target | Action |
|--------|--------|
| `make agent-sandbox` | A contained session: `AGENT=claude|opencode NAME=<task> [PROMPT="..."] [EFFORT=low] [CLAUDE_MODEL=...] [MODEL=...]` |
| `make agent-sandbox-build` | Rebuild both images; required after editing `Dockerfile.agents`, `proxy/*`, `uv.lock`, or `rust-harness/Cargo.lock` |
| `make agent-sandbox-clean` | Tear down the shared proxy container and the two networks; the inference stack is untouched |
| `make agent-versions` | Compare the CLI pins against the npm registry, check-only |
| `make agent-preflight` | Command-directory parity plus the supply-chain scan of both tool families |
| `make agent-export` | Export a session transcript: `AGENT=opencode|claude ID=<session-id>` |
| `make agent-run` | The uncontained batch rung for Claude Code on the host; it refuses OpenCode's `--auto` mode and points at the sandbox instead |

## Reference implementation

- `docker/agents/Dockerfile.agents` — three-stage build: uv source, Rust source with the
  prefetched registry, node base with both CLIs and the lockfile environment.
- `docker/agents/proxy/Dockerfile.proxy` and `docker/agents/proxy/squid.conf` — the pinned
  squid and its deny-by-default allowlist.
- `docker/agents/opencode.sandbox.json` — autoupdate and sharing off.
- `docker-compose.agents.yaml` — the two services, the two networks, the tmpfs mounts, and
  the limits.
- `make/agents.mk` — the wrapper, the build and teardown targets, and the seeded settings.
- [agent-sandbox.md](agent-sandbox.md) — the threat model these files implement, the egress
  allowlist's reasons, the credential rules, and the quarterly review checklist.
