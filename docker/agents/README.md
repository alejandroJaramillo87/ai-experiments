# docker/agents

The build context for the contained agent session: the agent image, its egress proxy, and
the OpenCode configuration the image copies in. The services (`agent`, `agent-proxy`),
their networks and the one scoped key the compose file injects are defined in
[../../docker-compose.agents.yaml](../../docker-compose.agents.yaml), which is the source
of truth; this directory is what that definition builds from.

## Layout

<!-- inventory:build:start -->

| File | Purpose |
|------|---------|
| `Dockerfile.agents` | Containment image for coding agents (OpenCode + Claude Code). |

<!-- inventory:build:end -->

[proxy/](proxy/) holds `Dockerfile.proxy`, the Squid egress proxy, and `squid.conf`, the
allowlist that is the agent's only path out. `opencode.sandbox.json` is the OpenCode
configuration the image ships: autoupdate and sharing off.

## Traps

**The image is coupled to the Rust workspace.** `Dockerfile.agents` bakes in the toolchain
pinned by `rust-harness/rust-toolchain.toml` and a prefetched crate registry, so
`make agent-sandbox-build` is required after changing that file, `rust-harness/Cargo.lock`,
`uv.lock`, or anything under this directory.

**Egress is the allowlist.** HTTPS is allowlisted by CONNECT host in `proxy/squid.conf`, with
no TLS interception, which is why `uv sync` cannot reach the pytorch index from inside a
session.

## Consumers

- `make agent-sandbox-build` builds both images; `make agent-sandbox` starts a session.
- [../../docs/sandbox/agent-sandbox.md](../../docs/sandbox/agent-sandbox.md) explains the
  containment model, and [../../docs/agents/](../../docs/agents/README.md) how the agents
  are run.
