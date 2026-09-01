"""Measure a step, following the protocol rather than reinventing it per run.

Three rules from `.claude/skills/benchmarking/SKILL.md`, in one place so a
measurement taken in a runbook entry is the same measurement taken six months
later:

  1. Discard the first run. Caches, allocator arenas and JIT are not settled,
     and on the served backends the first request pays connection setup.
  2. Take several samples and report the **median and the spread**, never a
     single best number. A best-of-N is a claim about the luckiest run.
  3. Record the fixed conditions beside the number, because a rate without its
     workload is not comparable to anything.

What this deliberately does not do is decide whether a number is good. The
regression criterion -- 5% off a recorded baseline, outside the spread -- needs
a baseline for this workload on this machine, and there is none until the
workstation runs one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

# Enough samples that a median means something, few enough that a benchmark is
# not an errand. The skill asks for at least three; the warm-up is on top.
DEFAULT_SAMPLES = 3

WHAT = ("ingest", "triage", "retrieval")


def configure(sub) -> None:
    p = sub.add_parser(
        "bench",
        help="measure ingest, triage, or retrieval: median and spread over N samples",
        description=__doc__,
    )
    p.add_argument("what", choices=list(WHAT), help="which step to measure")
    p.add_argument(
        "--log",
        default=None,
        help="mongod log to use for ingest and triage",
    )
    p.add_argument(
        "--store",
        default="memory",
        choices=["memory", "mongo"],
        help="store backend for ingest and triage (default: memory)",
    )
    p.add_argument(
        "--index",
        default=None,
        help="on-disk index for retrieval (default: <kb dir>/index)",
    )
    p.add_argument(
        "--uri", default=None, help="MongoDB connection string (default: the env var)"
    )
    p.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"measured runs after the discarded warm-up (default: {DEFAULT_SAMPLES})",
    )
    p.add_argument("--json", action="store_true", help="emit the result as JSON")
    p.set_defaults(func=cmd)


def _sample(work: Callable[[], int], samples: int, out) -> tuple[list[float], int]:
    """Run the work once to warm up, then `samples` times against the clock.

    The warm-up's result is used for the units line and then its timing is
    thrown away -- it is a real run, just not a measured one.
    """
    out.run("Warm-up run (discarded)")
    units = work()

    timings: list[float] = []
    for number in range(1, samples + 1):
        started = time.monotonic()
        work()
        elapsed = time.monotonic() - started
        timings.append(elapsed)
        out.info(f"sample {number}: {elapsed:.3f}s")
    return timings, units


def _ingest_work(args) -> Callable[[], int]:
    from logtriage.mongod.normalize import normalize
    from logtriage.mongod.records import LogRecord, read_log
    from logtriage.store.factory import log_store

    def work() -> int:
        store = log_store(args.store, args.uri)
        count = 0
        records = []
        for item in read_log(args.log):
            if isinstance(item, LogRecord):
                records.append(normalize(item))
                count += 1
        store.insert(records)
        return count

    return work


def _triage_work(args) -> Callable[[], int]:
    from logtriage.mongod.diagnostics import run_all
    from logtriage.mongod.normalize import normalize
    from logtriage.mongod.records import LogRecord, read_log
    from logtriage.store.factory import log_store
    from logtriage.triage.profile import build_profile

    def work() -> int:
        records = [
            normalize(item)
            for item in read_log(args.log)
            if isinstance(item, LogRecord)
        ]
        store = log_store(args.store, args.uri)
        store.insert(records)
        # Profile and detectors only. The model is deliberately outside the
        # measurement: a served model's latency swamps everything else and is
        # measured separately, against a served model.
        build_profile(records, source=args.log)
        run_all(store)
        return len(records)

    return work


def _retrieval_work(args, out) -> Callable[[], int]:
    from logtriage.commands.triage import _load_kb
    from logtriage.evals.metrics import load

    store, embedder = _load_kb(args.index or "", out)
    questions = load()

    def work() -> int:
        for question in questions:
            (vector,) = embedder.embed([question.question])
            store.hybrid_search(question.question, vector, k=10)
        return len(questions)

    return work


def cmd(args: argparse.Namespace) -> int:
    from logtriage.cli import LogtriageError
    from logtriage.output import Printer

    out = Printer(quiet=args.json)

    if args.what in ("ingest", "triage") and not args.log:
        raise LogtriageError(
            f"bench {args.what} needs a log to measure\n"
            "  Solution: point it at one, or generate one\n"
            "  Command: python -m logtriage gen-logs --out mongod.log"
        )
    if args.samples < 1:
        raise LogtriageError("--samples must be at least 1")

    if args.what == "ingest":
        work, unit = _ingest_work(args), "records"
    elif args.what == "triage":
        work, unit = _triage_work(args), "records"
    else:
        work, unit = _retrieval_work(args, out), "questions"

    timings, units = _sample(work, args.samples, out)

    median = statistics.median(timings)
    spread = max(timings) - min(timings)
    payload = {
        "what": args.what,
        "samples": args.samples,
        "conditions": {
            "log": args.log,
            "store": args.store,
            "index": args.index,
            unit: units,
        },
        "median_seconds": round(median, 4),
        "spread_seconds": round(spread, 4),
        "min_seconds": round(min(timings), 4),
        "max_seconds": round(max(timings), 4),
        f"{unit}_per_second": round(units / median) if median > 0 else None,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    out.line()
    out.line(f"  {args.what}: {units:,} {unit}, {args.store} store")
    out.line(f"  median         {median:.3f}s")
    out.line(
        f"  spread         {spread:.3f}s  (min {min(timings):.3f}, max {max(timings):.3f})"
    )
    if median > 0:
        out.line(f"  rate           {round(units / median):,} {unit}/s")
    out.line()

    # A spread wider than the median means the samples disagree by more than the
    # thing being measured, and comparing that against a baseline is comparing
    # noise. Said here rather than left for a reader to notice.
    if spread > median:
        out.warn(
            "The spread exceeds the median: these samples are not comparable to "
            "anything. Re-run on an idle machine, or take more samples."
        )
    out.ok(f"Measured {args.what} over {args.samples} samples after a warm-up")
    return 0
