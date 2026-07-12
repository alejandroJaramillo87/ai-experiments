# preflight fixture (planted supply-chain vectors)

A synthetic "third-party checkout" used to prove `make agent-preflight` flags
both agent config families. It contains two deliberately hostile, inert decoys:

- `.opencode/plugin/evil.ts` — an auto-loaded OpenCode plugin (runs in-process
  with full privileges at startup).
- `.claude/settings.json` — a Claude Code hook that executes a command on every
  tool use.

Neither is ever loaded by this repo (they live under `tests/`, not at the repo
root). The acceptance check is:

```bash
make agent-preflight PREFLIGHT_DIR=tests/agents/preflight_fixture
```

which must flag both vectors and exit non-zero. See
`scripts/agents/preflight-scan.sh` and `docs/sandbox/agent-sandbox.md`.
