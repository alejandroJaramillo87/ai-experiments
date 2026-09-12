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
// A harness chat run's envelope: RunHeader::fields() merged on the shared
// keys, the way rust-harness/obs writes it. Its page has two charts and a
// strip that states model, fold and build (RHX.4).
const HARNESS_RUN_ID = "R-20260912-160320-91dad9";
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

function writeHarnessFixtures(base) {
  const runDir = path.join(base, "run-harness");
  const eventsDir = path.join(base, "obs-logs", "harness");
  fs.mkdirSync(runDir, { recursive: true });
  fs.mkdirSync(eventsDir, { recursive: true });
  const header = {
    schema_version: 1,
    report: "run_metrics",
    timestamp: "2026-09-12T16:03:26.184+00:00",
    run_id: HARNESS_RUN_ID,
    cli: "harness",
    command: "chat",
    model: "/models/hf/gemma-4-E4B",
    model_hash: "e376216f8b693aff",
    n_layers: 42,
    dtype: "bf16",
    threads: 8,
    dot: "widen4",
    batch: "64",
    binary_version: "0.1.0",
    git_sha: "unknown",
  };
  const metrics = [header];
  // rss_gb flat within tens of KB, the workstation's shape: the first
  // 2026-09-12 page drew six labels all reading 14.242, and the second, with
  // the precision fixed, drew 14.24200..14.24240 clipped to their last five
  // digits by the default 50 px gutter. A 0.0001 spacing gave labels one
  // character shorter that fit, so it did not reproduce the clipping.
  for (let step = 1; step <= 8; step += 1) {
    metrics.push({
      step,
      ts: `2026-09-12T16:03:3${step}+00:00`,
      tokens_per_s: step / 2,
      rss_gb: 14.242 + step * 0.00001,
    });
  }
  fs.writeFileSync(
    path.join(runDir, "metrics.jsonl"),
    metrics.map((r) => JSON.stringify(r)).join("\n") + "\n"
  );
  fs.writeFileSync(
    path.join(eventsDir, `${HARNESS_RUN_ID}.jsonl`),
    JSON.stringify({
      ts: "2026-09-12T16:03:26+00:00",
      level: "info",
      event: "run.header",
      msg: "resolved run configuration",
    }) + "\n"
  );
  return runDir;
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
  check(
    uplots === 3,
    `charts: ${uplots} uPlot instances (loss, lr, throughput)`
  );
  const shown = await page.locator(".chart:visible").count();
  check(shown === 3, `charts: ${shown} holders shown, the empty groups hidden`);
  const strip = await page.locator(".header").innerText();
  check(
    strip.includes("tier") &&
      strip.includes("sft") &&
      strip.includes("test/model"),
    "header: the training strip states tier and model"
  );

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
  check(
    errors.length === 0,
    `no page errors (${errors.join("; ") || "clean"})`
  );
}

async function harnessChecks(page, htmlPath) {
  console.log("[RUN] harness snapshot page (file://)");
  const errors = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  await page.goto(`file://${htmlPath}`);
  await page.waitForFunction(
    () => document.getElementById("root").__obsState !== undefined
  );

  const uplots = await page.locator(".uplot").count();
  check(uplots === 2, `charts: ${uplots} uPlot instances (throughput, memory)`);
  const shown = await page.locator(".chart:visible").count();
  check(
    shown === 2,
    `charts: ${shown} holders shown, four empty groups hidden`
  );

  const items = await page.evaluate(() =>
    Array.from(document.querySelectorAll(".hdr-item")).map((n) => [
      n.querySelector(".hdr-key").textContent,
      n.querySelector(".hdr-val").textContent,
    ])
  );
  const strip = Object.fromEntries(items);
  check(
    strip.model === "/models/hf/gemma-4-E4B" &&
      strip.hash === "e376216f8b693aff" &&
      strip.fold === "dot=widen4 batch=64 threads=8" &&
      strip.build === "0.1.0 git unknown",
    `header: the harness strip states model, hash, fold and build (${JSON.stringify(strip)})`
  );
  check(
    !("tier" in strip) && !Object.values(strip).includes("-"),
    "header: no training key and no dash on a harness page"
  );

  // The labels uPlot drew for the memory axis, read off the axis it kept.
  const labels = await page.evaluate(() => {
    const u =
      document.getElementById("root").__obsState.charts["Process memory (GB)"]
        .plot;
    return u.axes[1]._values;
  });
  check(
    Array.isArray(labels) &&
      labels.length > 1 &&
      new Set(labels).size === labels.length,
    `memory axis: ${labels.length} labels, all distinct (${labels.join(" ")})`
  );
  // The gutter must hold the widest label plus the gap and the tick, or the
  // label's leading digits fall off the canvas.
  const gutter = await page.evaluate(() => {
    const u =
      document.getElementById("root").__obsState.charts["Process memory (GB)"]
        .plot;
    const axis = u.axes[1];
    u.ctx.font = axis.font[0];
    const widest = Math.max(
      ...axis._values.map((v) => u.ctx.measureText(v).width)
    );
    return {
      size: axis._size,
      needed: widest / (uPlot.pxRatio || 1) + axis.gap + axis.ticks.size,
    };
  });
  check(
    gutter.size >= gutter.needed,
    `memory axis: gutter ${gutter.size.toFixed(1)} px holds the widest label ` +
      `(${gutter.needed.toFixed(1)} px needed)`
  );
  check(
    errors.length === 0,
    `no page errors (${errors.join("; ") || "clean"})`
  );
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
  const harnessHtml = buildSnapshot(writeHarnessFixtures(base), base);
  ok(`fixtures and snapshots built under ${base}`);

  const browser = await chromium.launch({
    executablePath: findChromium(),
    args: ["--no-sandbox", "--disable-gpu"],
  });
  let server = null;
  try {
    const page = await browser.newPage();
    await snapshotChecks(page, htmlPath, eventCount);
    console.log();
    await harnessChecks(await browser.newPage(), harnessHtml);
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
