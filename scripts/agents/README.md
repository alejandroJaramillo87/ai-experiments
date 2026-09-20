# scripts/agents

The guardrails and checks behind the agent tooling: the tool-call guard hook, the
supply-chain preflight, the OpenCode agent generator, the role acceptance tests and the
version and idle checks. Everything here runs in a sandbox except the acceptance tests,
which spend tokens against a live model.

Two scripts are wired from `.claude/settings.json` rather than from `make`: `guard.sh` is
the `PreToolUse` hook that blocks a tool call the permission patterns cannot see, such as
a secrets-file read through shell indirection, and `runpod-idle-check.sh` is the `Stop`
and `SessionEnd` hook that warns about a warm RunPod worker. `gen_opencode_agents.py` is
the reason `.opencode/agents/` is never edited by hand: `make agent-sync` regenerates it
from `.claude/agents/` and `make agent-parity` fails when the two disagree. The
deterministic half of all of this is tested under `tests/agents/`; the acceptance tests
are run with `make agent-role-tests`.

## Scripts

<!-- inventory:scripts:start -->

| Script | Purpose |
|--------|---------|
| `check-agent-versions.sh` | Compare the CLI version pins in docker/agents/Dockerfile.agents against the npm registry. |
| `gen_opencode_agents.py` | Generate .opencode/agents/*.md from .claude/agents/*.md. |
| `guard.sh` | Single-source tool-call guardrail for both OpenCode and Claude Code. |
| `preflight-scan.sh` | Supply-chain scan for coding-agent config. |
| `role-acceptance.sh` | LLM acceptance tests for the QA subagent roles. |
| `runpod-idle-check.sh` | "never pay for a forgotten warm worker" guardrail. |

<!-- inventory:scripts:end -->

## Where the real docs live

- [../../docs/agents/README.md](../../docs/agents/README.md) — the agent tooling, the roster and the ladder
- [../../docs/agents/tooling-parity.md](../../docs/agents/tooling-parity.md) — what is single-sourced between the two tools and how
- [../../tests/agents/README.md](../../tests/agents/README.md) — the fixtures these scripts are tested against
