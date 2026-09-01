"""Read a mongod log into a store, and report how fast it went.

The verb that was missing. `MongoLogStore` has existed since the served stack
was written and nothing could reach it: every other command builds an in-memory
store from a file it just parsed, so the aggregation path was code with no
caller.

The timing is not a separate stopwatch bolted on. Ingest is the one step whose
cost scales with the log rather than with the corpus summary, so it is the step
worth measuring, and a command that reports its own rate is the only version of
that measurement nobody has to remember to take.

Records stream. `read_log` yields, normalization is a generator, and the store
batches -- so a 100MB log is never materialised as a list. That matters for the
number as much as for the memory: a rate measured while the machine is swapping
is a rate that says nothing about the code.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from typing import Any

from logtriage.store.factory import BACKENDS


def configure(sub) -> None:
    p = sub.add_parser(
        "ingest",
        help="read a mongod log into a store, reporting the ingest rate",
        description=__doc__,
    )
    p.add_argument("log", help="path to a mongod JSON log file")
    p.add_argument(
        "--store",
        default="memory",
        choices=list(BACKENDS),
        help="where to write (default: memory, which is a dry run that reports "
        "the same counts and is discarded when the process exits)",
    )
    p.add_argument(
        "--uri",
        default=None,
        help="MongoDB connection string (default: LOGTRIAGE_MONGO_URI)",
    )
    p.add_argument(
        "--ensure-indexes",
        action="store_true",
        help="create the indexes the queries assume before writing; idempotent, "
        "and a facet without them reads every document",
    )
    p.add_argument(
        "--redact",
        action="store_true",
        help="replace literal command values with type placeholders at ingest",
    )
    p.add_argument("--json", action="store_true", help="emit the result as JSON")
    p.set_defaults(func=cmd)


def _normalized(parsed, redact: bool, counts: dict[str, int]) -> Iterator[Any]:
    """Normalize as the parser yields, counting both outcomes on the way past.

    A generator rather than a list comprehension so the corpus is never held
    whole. The counters are a dict rather than returned values because a
    generator's totals are only known once it has been consumed, and the
    consumer here is the store.
    """
    from logtriage.mongod.normalize import normalize
    from logtriage.mongod.records import LogRecord

    for item in parsed:
        if isinstance(item, LogRecord):
            counts["records"] += 1
            yield normalize(item, redact=redact)
        else:
            counts["parse_errors"] += 1


def cmd(args: argparse.Namespace) -> int:
    from pathlib import Path

    from logtriage.cli import LogtriageError
    from logtriage.mongod.records import read_log
    from logtriage.output import Printer
    from logtriage.store.factory import log_store

    out = Printer(quiet=args.json)

    path = Path(args.log)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise LogtriageError(
            f"Cannot read log: {exc}\n"
            f"  Solution: check the path exists and is readable\n"
            f"  Command: ls -l {args.log}"
        ) from exc

    store = log_store(args.store, args.uri)

    if args.ensure_indexes:
        ensure = getattr(store, "ensure_indexes", None)
        if ensure is None:
            # Not an error. The in-memory store scans, so there is nothing to
            # create, and refusing the flag would make the two backends need
            # different command lines for the same job.
            out.skip(f"--ensure-indexes: the {args.store} store has no indexes")
        else:
            out.run("Creating indexes")
            ensure()

    counts = {"records": 0, "parse_errors": 0}
    out.run(f"Ingesting {path} ({size:,} bytes) into the {args.store} store")

    started = time.monotonic()
    try:
        written = store.insert(_normalized(read_log(args.log), args.redact, counts))
    except OSError as exc:
        raise LogtriageError(f"Cannot read log: {exc}") from exc
    elapsed = time.monotonic() - started

    if not counts["records"]:
        raise LogtriageError(
            f"No mongod log records found in {args.log} "
            f"({counts['parse_errors']} unparseable lines)\n"
            "  Solution: confirm this is a mongod 4.4+ structured (JSON) log\n"
            f"  Command: head -1 {args.log}"
        )

    # Guarded: a fixture small enough to ingest inside the clock's resolution
    # would otherwise divide by zero, and reporting an infinite rate for a fast
    # run is worse than reporting none.
    rate = counts["records"] / elapsed if elapsed > 0 else None

    payload = {
        "log": str(path),
        "store": args.store,
        "bytes": size,
        "records": counts["records"],
        "written": written,
        "parse_errors": counts["parse_errors"],
        "seconds": round(elapsed, 4),
        "records_per_second": round(rate) if rate else None,
        "bytes_per_second": round(size / elapsed) if elapsed > 0 else None,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    out.line()
    out.line(f"  records        {counts['records']:,}")
    out.line(f"  written        {written:,}")
    out.line(f"  parse errors   {counts['parse_errors']:,}")
    out.line(f"  elapsed        {elapsed:.2f}s")
    if rate:
        out.line(
            f"  rate           {round(rate):,} records/s, {size / elapsed / 1e6:.1f} MB/s"
        )
    out.line()

    if counts["parse_errors"]:
        # Evidence that was never read changes how much a conclusion is worth,
        # so this is a warning even though the ingest succeeded.
        out.warn(f"Unparseable lines: {counts['parse_errors']:,}")

    out.ok(f"Ingested {written:,} records into the {args.store} store")
    if args.store == "memory":
        out.warn(
            "The memory store is discarded when this process exits; nothing was "
            "persisted. Use --store mongo to keep it."
        )
    return 0
