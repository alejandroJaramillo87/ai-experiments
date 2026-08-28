"""Probe monitor (W4): aggregate the scores the serving process already took.

Reads a window of served-request logs and reports each probe's score
distribution, the records over threshold, and the drift versus the previous
run over the same probe set.

It does not run a model. Every record carries the probe scores computed in the
process that answered the request, by the tap seam in
`rust-harness/harness/src/interplog.rs`, so there is nothing left to replay.
That is the whole difference from the stage this replaced: llama-server had no
activation read path, so scoring meant re-running a second copy of the model
offline over a request log a proxy had captured. Three things went with the
replay -- the proxy, the GPU contention that made this batch work, and the
fidelity caveat, which existed because the replay model was bf16 while the
serving model was quantized. Scores now come from the weights that produced
the text, so there is no gap left to describe.

Output: one report JSON per run under the monitor directory plus flagged.jsonl
beside it (records over threshold -- alerting is a file, not a pager).
Malformed log records are counted as skipped, never a crash.
"""

from __future__ import annotations

import datetime
import json
import os
import re

import numpy as np

from interp.manifest import git_sha, text_sha256, utc_now

DEFAULT_LOG_ROOT = "/mnt/ai-data/interp/logs"
DEFAULT_MONITOR_ROOT = "/mnt/ai-data/interp/monitor"
RETENTION_DAYS = 30

# Reports are comparable only against reports produced the same way. Recorded
# in every report and matched by `previous_report`, so a drift delta cannot be
# computed across a change in how the score was obtained.
SCORED_IN_BAND = "in_band"


class MonitorError(Exception):
    """Preconditions failed: no probes, or an empty window."""


def parse_window(window: str) -> datetime.timedelta:
    match = re.fullmatch(r"(\d+)([dh])", window)
    if not match:
        raise MonitorError(f"bad --window {window!r}: use forms like 7d or 12h")
    value, unit = int(match.group(1)), match.group(2)
    return datetime.timedelta(**{"days" if unit == "d" else "hours": value})


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
    """Rebuild the text a record describes, for the report's top-record ids.

    No longer feeds a scorer -- the score is in the record. It is still what
    identifies a record to a person reading the report, and the id is a hash
    of it, so a record whose text cannot be rebuilt is skipped rather than
    reported under an id that means nothing.
    """
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


def probe_score(record: dict) -> tuple[str, float, float] | None:
    """The (probe name, score, threshold) one record carries, or None.

    The score is the maximum over the request's scored positions, which is the
    statistic the serving process records: one number per request, the same
    granularity the report has always been built from.

    `None` covers a record served without a probe loaded, which is ordinary
    (the endpoint answers whether or not interp asked it to observe) and is
    counted as skipped rather than treated as a zero. A zero would drag every
    distribution toward it.
    """
    probe = record.get("probe")
    if not isinstance(probe, dict):
        return None
    name = probe.get("name")
    score = probe.get("max_score")
    threshold = probe.get("threshold")
    if not isinstance(name, str):
        return None
    if not isinstance(score, (int, float)) or not isinstance(threshold, (int, float)):
        return None
    return name, float(score), float(threshold)


def read_log_window(
    log_dir: str,
    since: datetime.datetime,
    until: datetime.datetime,
) -> tuple[list[dict], dict]:
    """Load records in [since, until); count skips by reason, never crash."""
    records: list[dict] = []
    skipped = {
        "malformed_json": 0,
        "no_timestamp": 0,
        "out_of_window": 0,
        "no_text": 0,
        "no_scores": 0,
    }
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
                scored = probe_score(record)
                if scored is None:
                    skipped["no_scores"] += 1
                    continue
                record["_text"] = text
                record["_id"] = text_sha256(record["timestamp"] + text)[:16]
                record["_probe"], record["_score"], record["_threshold"] = scored
                records.append(record)
    return records, skipped


def previous_report(
    monitor_dir: str, probe_names: list[str], scored: str = SCORED_IN_BAND
) -> dict | None:
    """Most recent prior report over the same probe set, for the drift delta.

    Matched on `scored` as well as the probe set. A report built from scores
    the serving process took and one built by replaying a second model are two
    measurements of different things; subtracting them would present the
    change in method as drift in the model.
    """
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
        if report.get("scored") != scored:
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
    log_dir: str,
    monitor_root: str = DEFAULT_MONITOR_ROOT,
    window: str = "7d",
    probes: list[str] | None = None,
    max_records: int | None = None,
    top_n: int = 20,
    now: datetime.datetime | None = None,
) -> dict:
    """Aggregate a log window into a report plus flagged.jsonl.

    `probes` filters by name; omitted, every probe appearing in the window is
    reported. There is no probe artifact to load: the record names its probe
    and carries the threshold that probe was scored against, and the serving
    process already refused any probe fitted on weights other than the ones it
    serves. Checking the pairing again here would re-derive a weaker version
    of a guarantee that has already been enforced at the point it matters.
    """
    now = now or datetime.datetime.now(datetime.UTC)
    since = now - parse_window(window)
    # Read the window before enforcing retention: a --window longer than the
    # retention period must replay what is still on disk, not delete it
    # first and then silently omit it
    records, skipped = read_log_window(log_dir, since, now)
    removed = enforce_retention(log_dir, now=now)
    total = len(records) + sum(skipped.values())
    if probes:
        wanted = set(probes)
        dropped_by_filter = [r for r in records if r["_probe"] not in wanted]
        records = [r for r in records if r["_probe"] in wanted]
        skipped["other_probe"] = len(dropped_by_filter)
    dropped = 0
    if max_records is not None and len(records) > max_records:
        # newest-first up to the cap; say what was dropped (W4 decision 5)
        records.sort(key=lambda r: r["timestamp"], reverse=True)
        dropped = len(records) - max_records
        records = records[:max_records]
    if not records:
        raise MonitorError(f"no scored records in window {window} under {log_dir}")

    by_probe: dict[str, list[dict]] = {}
    for record in records:
        by_probe.setdefault(record["_probe"], []).append(record)

    target_services = sorted(
        {
            r["target_service"]
            for r in records
            if isinstance(r.get("target_service"), str)
        }
    )

    monitor_dir = monitor_root
    run_name = now.strftime("%Y-%m-%dT%H%M%SZ")
    run_dir = os.path.join(monitor_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    prior = previous_report(monitor_dir, sorted(by_probe))
    report_probes = {}
    flagged_path = os.path.join(run_dir, "flagged.jsonl")
    flagged_count = 0
    with open(flagged_path, "w", encoding="utf-8") as flagged_file:
        for name, probe_records in sorted(by_probe.items()):
            probe_scores = np.array([r["_score"] for r in probe_records], dtype=float)
            # The threshold each record was scored against. They agree unless
            # the served probe was re-exported mid-window; the maximum is the
            # conservative reading, and the disagreement is reported rather
            # than averaged away.
            thresholds = sorted({r["_threshold"] for r in probe_records})
            threshold = max(thresholds)
            dist = distribution(probe_scores, threshold)
            dist["threshold"] = threshold
            if len(thresholds) > 1:
                dist["thresholds_observed"] = thresholds
            order = np.argsort(-probe_scores)[:top_n]
            dist["top_records"] = [
                {
                    "id": probe_records[int(i)]["_id"],
                    "score": round(float(probe_scores[int(i)]), 4),
                }
                for i in order
            ]
            if prior:
                prior_dist = prior["probes"].get(name, {})
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
            report_probes[name] = dist
            for i in np.flatnonzero(probe_scores > threshold):
                flagged_file.write(
                    json.dumps(
                        {
                            "probe": name,
                            "id": probe_records[int(i)]["_id"],
                            "score": round(float(probe_scores[int(i)]), 4),
                            "timestamp": probe_records[int(i)]["timestamp"],
                        }
                    )
                    + "\n"
                )
                flagged_count += 1

    report = {
        "schema_version": 2,
        "kind": "probe_monitor",
        "run": run_name,
        "timestamp": utc_now(),
        "git_sha": git_sha(),
        "scored": SCORED_IN_BAND,
        "window": window,
        "log_dir": log_dir,
        "counts": {
            "total": total,
            "reported": len(records),
            "skipped": skipped,
            "dropped_over_cap": dropped,
            "flagged": flagged_count,
            "retention_removed_files": removed,
        },
        "probes": report_probes,
        "target_services": target_services,
        "note": (
            "scores were taken in the process that answered each request, so "
            "they describe the model that produced the text; there is no "
            "replay and no precision gap to state"
        ),
    }
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    report["out_dir"] = run_dir
    return report
