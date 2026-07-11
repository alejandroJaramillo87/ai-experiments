// guard.ts - OpenCode adapter for the shared guardrail.
//
// Thin wrapper: on tool.execute.before it normalizes OpenCode's (tool, args) into the
// Claude-hook shape {tool_name, tool_input} and pipes it to scripts/agents/guard.sh, the
// single source of truth shared with Claude Code. A non-zero exit throws to block the call.
//
// Dependency-light: uses only Node's child_process (bun ships it); no npm installs.

import { spawnSync } from "node:child_process";

const SCRIPT = "scripts/agents/guard.sh";

export const GuardPlugin = async ({ directory }: { directory: string }) => {
  return {
    "tool.execute.before": async (
      input: { tool: string; sessionID: string; callID: string },
      output: { args: Record<string, unknown> },
    ) => {
      const payload = JSON.stringify({
        tool_name: input.tool,
        tool_input: output.args ?? {},
      });

      const res = spawnSync("bash", [SCRIPT], {
        input: payload,
        cwd: directory,
        encoding: "utf8",
      });

      if (res.status !== 0) {
        const reason = (res.stderr || "blocked by guard.sh").trim();
        // Throwing aborts the tool call; the message surfaces to the model.
        throw new Error(reason);
      }
    },
  };
};

export default GuardPlugin;
