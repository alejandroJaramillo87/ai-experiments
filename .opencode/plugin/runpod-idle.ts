// runpod-idle.ts - OpenCode adapter for the "never pay for a forgotten warm worker" check.
//
// On session.idle it runs scripts/agents/runpod-idle-check.sh (shared with Claude Code's
// Stop/SessionEnd hook). Any [WARN] line the script prints is surfaced as a toast so a warm
// endpoint is not silently left billing.
//
// Dependency-light: Node child_process only.

import { spawnSync } from "node:child_process";

const SCRIPT = "scripts/agents/runpod-idle-check.sh";

export const RunpodIdlePlugin = async ({
  directory,
  client,
}: {
  directory: string;
  client?: { tui?: { showToast?: (opts: { message: string; variant?: string }) => void } };
}) => {
  return {
    "session.idle": async () => {
      const res = spawnSync("bash", [SCRIPT], { cwd: directory, encoding: "utf8" });
      const out = `${res.stdout ?? ""}`;
      const warnings = out
        .split("\n")
        .filter((line) => line.includes("[WARN] RunPod:"))
        .map((line) => line.replace("[WARN] RunPod:", "RunPod:").trim());

      for (const w of warnings) {
        client?.tui?.showToast?.({ message: w, variant: "warning" });
      }
    },
  };
};

export default RunpodIdlePlugin;
