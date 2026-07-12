// usage.ts - OpenCode-only spend observability.
//
// Accumulates per-session token counts by provider + variant into the usage state file that
// `make agent-usage` reads (default .agent-state/usage.jsonl, override AGENT_USAGE_STATE).
// Claude Code has no counterpart here: it uses the built-in /usage plus OpenTelemetry export
// (see docs/tooling-parity.md); make agent-usage merges both sides into one report.
//
// Written from scratch to stay dependency-light. A community plugin (opencode-token-monitor)
// can be wrapped instead if richer accounting is needed later — wrap, do not fork.

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

function stateFile(directory: string): string {
  const rel = process.env.AGENT_USAGE_STATE ?? ".agent-state/usage.jsonl";
  return resolve(directory, rel);
}

export const UsagePlugin = async ({ directory }: { directory: string }) => {
  return {
    // Fires after each assistant turn completes; record whatever token usage is reported.
    event: async ({ event }: { event: { type: string; properties?: any } }) => {
      if (event.type !== "message.updated" && event.type !== "session.idle") return;
      const info = event.properties?.info ?? event.properties;
      const tokens = info?.tokens;
      if (!tokens) return;

      const record = {
        sessionID: info.sessionID ?? info.id ?? "unknown",
        provider: info.providerID ?? info.provider ?? "unknown",
        variant: info.modelID ?? info.model ?? "unknown",
        input: tokens.input ?? 0,
        output: tokens.output ?? 0,
        reasoning: tokens.reasoning ?? 0,
      };

      const file = stateFile(directory);
      mkdirSync(dirname(file), { recursive: true });
      appendFileSync(file, JSON.stringify(record) + "\n");
    },
  };
};

export default UsagePlugin;
