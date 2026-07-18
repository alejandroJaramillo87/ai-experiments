/*
 * check.mjs — real-browser verification for the run dashboard (OBS-11/12).
 *
 * jsdom-free checks that only a browser can make: uPlot chart creation,
 * marker ingestion, log filtering, the hostile "<!--<script" payload, and
 * the live page extending from a running serve process. Fixtures are
 * generated fresh into a temp dir; the snapshot page is built by the real
 * report subcommand and loaded over file://, then the live page over the
 * real HTTP server. All waits go through waitForFunction — no sleeps.
 *
 * Usage: npm install && node check.mjs
 * Env: PW_CHROMIUM overrides the Chromium binary path.
 */

import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright-core";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "../../..");
const RUN_ID = "R-20260718-000000-b405e5";
const POLL_TIMEOUT_MS = 15000; // live page polls every 2s; several cycles

let failures = 0;

function ok(msg) {
  console.log(`[OK] ${msg}`);
}

function fail(msg) {
  failures += 1;
  console.log(`[FAIL] ${msg}`);
}

function check(condition, msg) {
  if (condition) ok(msg);
  else fail(msg);
}

function findChromium() {
  if (process.env.PW_CHROMIUM) return process.env.PW_CHROMIUM;
  const root = "/opt/pw-browsers";
  if (fs.existsSync(root)) {
    for (const name of fs.readdirSync(root)) {
      if (!name.startsWith("chromium")) continue;
      const candidate = path.join(root, name, "chrome-linux", "chrome");
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  console.log("[FAIL] No Chromium binary found");
  console.log("  Solution: set PW_CHROMIUM to a Chromium/Chrome executable");
  console.log("  Command: PW_CHROMIUM=$(command -v chromium) node check.mjs");
  process.exit(1);
}

function writeFixtures(base) {
  const runDir = path.join(base, "run");
  const eventsDir = path.join(base, "obs-logs", "training");
  fs.mkdirSync(runDir, { recursive: true });
  fs.mkdirSync(eventsDir, { recursive: true });
  const header = {
    report: "run_metrics",
    schema_version: 1,
    generated_at: "2026-07-18T00:00:00+00:00",
    git_sha: "0000000",
    run_id: RUN_ID,
    cli: "training",
    command: "train-sft",
    tier: "sft",
    model_id: "test/model",
    recipe_hash: "b405e5cafe",
    schedule: { type: "cosine", warmup_steps: 2, total_steps: 10 },
  };
  const metrics = [header];
  for (let step = 1; step <= 6; step += 1) {
    metrics.push({
      step,
      ts: `2026-07-18T00:00:0${step}+00:00`,
      loss: 2.0 / step,
      lr: 0.001 * step,
      tokens_per_s: 100 + step,
    });
  }
  metrics.push({ step: 3, checkpoint_saved: 1 });
  metrics.push({ step: 5, "watchdog/loss_spike": 1 });
  fs.writeFileSync(
    path.join(runDir, "metrics.jsonl"),
    metrics.map((r) => JSON.stringify(r)).join("\n") + "\n"
  );
  const event = (level, msg) => ({
    ts: "2026-07-18T00:00:00+00:00",
    level,
    event: "log",
    msg,
  });
  const events = [
    event("info", "startup complete"),
    event("info", "needle-alpha unique row"),
    event("warning", "disk almost full"),
    event("critical", "unknown level stays visible"),
    event("info", "worse <!--<script x payload"),
  ];
  fs.writeFileSync(
    path.join(eventsDir, `${RUN_ID}.jsonl`),
    events.map((r) => JSON.stringify(r)).join("\n") + "\n"
  );
  return { runDir, eventsDir, eventCount: events.length };
}

function buildSnapshot(runDir, base) {
  execFileSync(
    "uv",
    ["run", "--no-sync", "python", "-m", "training", "report", "--run", runDir],
    {
      cwd: REPO_ROOT,
      env: { ...process.env, OBS_LOG_DIR: path.join(base, "obs-logs") },
      stdio: ["ignore", "ignore", "inherit"],
    }
  );
  return path.join(runDir, "run.html");
}

async function snapshotChecks(page, htmlPath, eventCount) {
  console.log("[RUN] snapshot page (file://)");
  const errors = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  await page.goto(`file://${htmlPath}`);
  await page.waitForFunction(
    () => document.getElementById("root").__obsState !== undefined
  );

  const uplots = await page.locator(".uplot").count();
  check(uplots === 3, `charts: ${uplots} uPlot instances (loss, lr, throughput)`);

  const markers = await page.evaluate(
    () => document.getElementById("root").__obsState.markers
  );
  check(
    markers.length === 2 &&
      markers.some((m) => m.kind === "checkpoint" && m.step === 3) &&
      markers.some((m) => m.kind === "watchdog" && m.step === 5),
    "markers: checkpoint at step 3 and watchdog at step 5 ingested"
  );

  const rows = await page.locator(".log-body .log-row").count();
  check(rows === eventCount, `log: all ${eventCount} rows render by default`);

  await page.fill(".log-controls input", "needle-alpha");
  await page.waitForFunction(
    () => document.querySelectorAll(".log-body .log-row").length === 1
  );
  ok("log: search filter narrows to the one matching row");
  await page.fill(".log-controls input", "");

  await page.selectOption(".log-controls select", "warning");
  await page.waitForFunction(
    () => document.querySelectorAll(".log-body .log-row").length === 2
  );
  ok("log: warning filter keeps the warning and the unknown-level row");

  await page.selectOption(".log-controls select", "debug");
  await page.waitForFunction(
    () => document.querySelectorAll(".log-body .log-row").length > 2
  );
  const hostile = await page.evaluate(() =>
    Array.from(document.querySelectorAll(".log-row")).some((r) =>
      r.textContent.includes("<!--<script")
    )
  );
  check(hostile, "hostile <!--<script payload renders as text, page intact");
  check(errors.length === 0, `no page errors (${errors.join("; ") || "clean"})`);
}

function startServer(runDir, base) {
  const proc = spawn(
    "uv",
    [
      "run",
      "--no-sync",
      "python",
      "-m",
      "training",
      "report",
      "--run",
      runDir,
      "--serve",
      "--port",
      "0",
    ],
    {
      cwd: REPO_ROOT,
      env: { ...process.env, OBS_LOG_DIR: path.join(base, "obs-logs") },
      stdio: ["ignore", "ignore", "pipe"],
    }
  );
  return new Promise((resolve, reject) => {
    let buffer = "";
    const timer = setTimeout(
      () => reject(new Error(`server did not print its URL: ${buffer}`)),
      30000
    );
    proc.stderr.on("data", (chunk) => {
      buffer += chunk.toString();
      const match = buffer.match(/http:\/\/[\d.]+:\d+\//);
      if (match) {
        clearTimeout(timer);
        resolve({ proc, url: match[0] });
      }
    });
    proc.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`server exited early (code ${code}): ${buffer}`));
    });
  });
}

async function liveChecks(page, url, runDir, eventsDir, eventCount) {
  console.log("[RUN] live page (report --serve)");
  await page.goto(url);
  await page.waitForFunction(
    () =>
      document.getElementById("root").__obsState !== undefined &&
      document.getElementById("root").__obsState.header !== null,
    undefined,
    { timeout: POLL_TIMEOUT_MS }
  );
  ok("live: header arrives via polling");

  const stepsBefore = await page.evaluate(
    () => document.getElementById("root").__obsState.steps.length
  );
  fs.appendFileSync(
    path.join(runDir, "metrics.jsonl"),
    JSON.stringify({ step: 7, loss: 0.25, lr: 0.007, tokens_per_s: 107 }) + "\n"
  );
  fs.appendFileSync(
    path.join(eventsDir, `${RUN_ID}.jsonl`),
    JSON.stringify({
      ts: "2026-07-18T00:00:09+00:00",
      level: "info",
      event: "log",
      msg: "live-appended row",
    }) + "\n"
  );
  await page.waitForFunction(
    (before) =>
      document.getElementById("root").__obsState.steps.length > before,
    stepsBefore,
    { timeout: POLL_TIMEOUT_MS }
  );
  ok("live: appended metrics extend the chart data");
  await page.waitForFunction(
    (count) =>
      document.querySelectorAll(".log-body .log-row").length === count + 1,
    eventCount,
    { timeout: POLL_TIMEOUT_MS }
  );
  ok("live: appended event appears in the log tail");
}

async function main() {
  console.log("=== Run dashboard browser verification ===");
  console.log();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "obs-browser-"));
  const { runDir, eventsDir, eventCount } = writeFixtures(base);
  const htmlPath = buildSnapshot(runDir, base);
  ok(`fixtures and snapshot built under ${base}`);

  const browser = await chromium.launch({
    executablePath: findChromium(),
    args: ["--no-sandbox", "--disable-gpu"],
  });
  let server = null;
  try {
    const page = await browser.newPage();
    await snapshotChecks(page, htmlPath, eventCount);
    console.log();
    server = await startServer(runDir, base);
    await liveChecks(page, server.url, runDir, eventsDir, eventCount);
  } finally {
    if (server) server.proc.kill("SIGINT");
    await browser.close();
    fs.rmSync(base, { recursive: true, force: true });
  }

  console.log();
  if (failures > 0) {
    console.log(`TESTS FAILED (${failures})`);
    return 1;
  }
  console.log("ALL CHECKS PASSED");
  return 0;
}

main().then(
  (code) => process.exit(code),
  (err) => {
    console.log(`[FAIL] ${err.message}`);
    process.exit(1);
  }
);
