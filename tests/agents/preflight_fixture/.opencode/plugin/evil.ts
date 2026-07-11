// Planted supply-chain fixture — NOT real, do not load.
//
// Represents an attacker-controlled OpenCode plugin auto-loaded from a cloned
// third-party repo. OpenCode installs and runs .opencode/plugin/*.ts in-process
// with full user privileges at startup, so its mere presence in an untrusted
// checkout is the finding. make agent-preflight PREFLIGHT_DIR=tests/agents/preflight_fixture
// must flag this file.
export const plugin = async () => ({
  "tool.execute.before": async () => {
    // A real attack would exfiltrate here; the fixture only needs to exist.
    // e.g. read ~/.ssh, POST to an attacker host — exactly what containment stops.
  },
})
