"""Async probe monitor (W4): batch replay of serving logs, not a daemon.

Reads a window of W3 request logs, reconstructs each request's text, runs a
single batched bf16 capture per replay through the workbench model, and scores
every requested registered probe via the W2 score path. Runs as batch work:
it refuses to start while an inference service is healthy (the GPU belongs to
serving during serving) — asynchronous replay is the design, not a compromise,
since llama-server has no activation read path.

Output: one report JSON per run under the monitor directory plus
flagged.jsonl beside it (records over threshold — alerting is a file, not a
pager). Malformed log records are counted as skipped, never a crash. The
report includes deltas versus the previous run over the same probe set — the
drift signal later phases build on.
"""

from __future__ import annotations

import datetime
import json
import os
import re

import numpy as np

from interp.manifest import git_sha, text_sha256, utc_now
from interp.probes.registry import Probe, load_probe
from interp.probes.score import score_texts

DEFAULT_LOG_ROOT = "/mnt/ai-data/interp/logs"
DEFAULT_MONITOR_ROOT = "/mnt/ai-data/interp/monitor"
RETENTION_DAYS = 30
HEALTH_URLS = (
    "http://127.0.0.1:8002/v1/health",  # llama-gpu
    "http://127.0.0.1:8003/health",  # vllm-gpu
)


class MonitorError(Exception):
    """Preconditions failed: services up, no probes, or empty window."""


def parse_window(window: str) -> datetime.timedelta:
    match = re.fullmatch(r"(\d+)([dh])", window)
    if not match:
        raise MonitorError(f"bad --window {window!r}: use forms like 7d or 12h")
    value, unit = int(match.group(1)), match.group(2)
    return datetime.timedelta(**{"days" if unit == "d" else "hours": value})


def services_up(health_urls: tuple[str, ...] = HEALTH_URLS) -> list[str]:
    """URLs of inference services currently answering their healthchecks."""
    import requests

    up = []
    for url in health_urls:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                up.append(url)
        except requests.RequestException:
            continue
    return up


def enforce_retention(
    log_dir: str, days: int = RETENTION_DAYS, now: datetime.datetime | None = None
) -> list[str]:
    """Delete daily log files older than the retention policy; return them."""
    now = now or datetime.datetime.now(datetime.UTC)
    cutoff = (now - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    removed: list[str] = []
    if not os.path.isdir(log_dir):
        return removed
    for name in sorted(os.listdir(log_dir)):
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.jsonl", name)
        if match and match.group(1) < cutoff:
            os.remove(os.path.join(log_dir, name))
            removed.append(name)
    return removed


def reconstruct_text(record: dict, include_response: bool = True) -> str | None:
    """Rebuild the text a replay scores from one W3 log record."""
    parts = []
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and content:
                parts.append(content)
    elif isinstance(record.get("prompt"), str):
        parts.append(record["prompt"])
    if include_response and isinstance(record.get("response_text"), str):
        parts.append(record["response_text"])
    return "\n".join(parts) if parts else None


def read_log_window(
    log_dir: str,
    since: datetime.datetime,
    until: datetime.datetime,
) -> tuple[list[dict], dict]:
    """Load records in [since, until); count skips by reason, never crash."""
    records: list[dict] = []
    skipped = {"malformed_json": 0, "no_timestamp": 0, "out_of_window": 0, "no_text": 0}
    if not os.path.isdir(log_dir):
        return records, skipped
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(log_dir, name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped["malformed_json"] += 1
                    continue
                try:
                    stamp = datetime.datetime.strptime(
                        record.get("timestamp", ""), "%Y-%m-%dT%H:%M:%S.%fZ"
                    ).replace(tzinfo=datetime.UTC)
                except ValueError:
                    skipped["no_timestamp"] += 1
                    continue
                if not since <= stamp < until:
                    skipped["out_of_window"] += 1
                    continue
                text = reconstruct_text(record)
                if text is None:
                    skipped["no_text"] += 1
                    continue
                record["_text"] = text
                record["_id"] = text_sha256(record["timestamp"] + text)[:16]
                records.append(record)
    return records, skipped


def previous_report(monitor_dir: str, probe_names: list[str]) -> dict | None:
    """Most recent prior report over the same probe set, for the drift delta."""
    if not os.path.isdir(monitor_dir):
        return None
    for name in sorted(os.listdir(monitor_dir), reverse=True):
        path = os.path.join(monitor_dir, name, "report.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if sorted(report.get("probes", {})) == sorted(probe_names):
            return report
    return None


def distribution(scores: np.ndarray, threshold: float) -> dict:
    return {
        "mean": round(float(scores.mean()), 4),
        "p50": round(float(np.percentile(scores, 50)), 4),
        "p95": round(float(np.percentile(scores, 95)), 4),
        "fraction_over_threshold": round(float((scores > threshold).mean()), 4),
        "count": int(scores.size),
    }


def run_monitor(
    probe_dirs: list[str],
    replay_model: str,
    replay_model_hash: str,
    log_dir: str,
    monitor_root: str = DEFAULT_MONITOR_ROOT,
    window: str = "7d",
    max_records: int | None = None,
    top_n: int = 20,
    capture_fn=None,
    health_check=services_up,
    now: datetime.datetime | None = None,
) -> dict:
    """Replay a log window, score every probe, write report + flagged.jsonl."""
    up = health_check()
    if up:
        raise MonitorError(
            f"inference services are up ({', '.join(up)}); the monitor is batch "
            "work that owns the GPU — stop serving first (make down)"
        )

    probes: list[Probe] = [
        load_probe(d, expect_model_hash=replay_model_hash) for d in probe_dirs
    ]
    if not probes:
        raise MonitorError("no probes given")

    now = now or datetime.datetime.now(datetime.UTC)
    since = now - parse_window(window)
    # Read the window before enforcing retention: a --window longer than the
    # retention period must replay what is still on disk, not delete it
    # first and then silently omit it
    records, skipped = read_log_window(log_dir, since, now)
    removed = enforce_retention(log_dir, now=now)
    total = len(records) + sum(skipped.values())
    dropped = 0
    if max_records is not None and len(records) > max_records:
        # newest-first up to the cap; say what was dropped (W4 decision 5)
        records.sort(key=lambda r: r["timestamp"], reverse=True)
        dropped = len(records) - max_records
        records = records[:max_records]
    if not records:
        raise MonitorError(f"no replayable records in window {window} under {log_dir}")

    texts = [r["_text"] for r in records]
    scores = score_texts(probes, texts, replay_model, capture_fn=capture_fn)

    monitor_dir = monitor_root
    run_name = now.strftime("%Y-%m-%dT%H%M%SZ")
    run_dir = os.path.join(monitor_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    prior = previous_report(monitor_dir, [p.name for p in probes])
    report_probes = {}
    flagged_path = os.path.join(run_dir, "flagged.jsonl")
    flagged_count = 0
    with open(flagged_path, "w", encoding="utf-8") as flagged_file:
        for probe in probes:
            probe_scores = scores[probe.name]
            dist = distribution(probe_scores, probe.threshold)
            order = np.argsort(-probe_scores)[:top_n]
            dist["top_records"] = [
                {
                    "id": records[int(i)]["_id"],
                    "score": round(float(probe_scores[int(i)]), 4),
                }
                for i in order
            ]
            if prior:
                prior_dist = prior["probes"].get(probe.name, {})
                if "mean" in prior_dist:
                    dist["delta_vs_previous"] = {
                        "mean": round(dist["mean"] - prior_dist["mean"], 4),
                        "fraction_over_threshold": round(
                            dist["fraction_over_threshold"]
                            - prior_dist["fraction_over_threshold"],
                            4,
                        ),
                        "previous_run": prior["run"],
                    }
            report_probes[probe.name] = dist
            for i in np.flatnonzero(probe_scores > probe.threshold):
                flagged_file.write(
                    json.dumps(
                        {
                            "probe": probe.name,
                            "id": records[int(i)]["_id"],
                            "score": round(float(probe_scores[int(i)]), 4),
                            "timestamp": records[int(i)]["timestamp"],
                        }
                    )
                    + "\n"
                )
                flagged_count += 1

    report = {
        "schema_version": 1,
        "kind": "probe_monitor",
        "run": run_name,
        "timestamp": utc_now(),
        "git_sha": git_sha(),
        "replay_model": replay_model,
        "replay_model_hash": replay_model_hash,
        "window": window,
        "log_dir": log_dir,
        "counts": {
            "total": total,
            "replayed": len(records),
            "skipped": skipped,
            "dropped_over_cap": dropped,
            "flagged": flagged_count,
            "retention_removed_files": removed,
        },
        "probes": report_probes,
        "note": (
            "bf16 replay activations approximate the quantized service's "
            "internals; quantifying the gap is the Phase 5 "
            "quantization-feature-survival track"
        ),
    }
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    report["out_dir"] = run_dir
    return report
