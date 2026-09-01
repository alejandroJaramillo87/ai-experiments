"""Unified logtriage CLI.

`python -m logtriage <subcommand>` is the single entry point. Each subcommand
lives in logtriage/commands/ and keeps its heavier imports inside its handler so
that parsing and --help stay fast and dependency-free -- a --help that imports
drain3 and pymongo is a --help nobody can run on a machine where those are not
installed yet.

Exit codes: 0 success, 1 failure, 2 invalid usage.
Output follows the same conventions as the parent repository's Python CLIs:
human status lines on stderr, stdout reserved for --json payloads.

Dispatch is local rather than shared. The parent repository routes its CLIs
through a traced dispatcher that stamps a run id and writes event records; this
project cannot import it and keep the property that it works standalone, so it
does the small part of that job it actually needs -- turn an expected failure
into a styled message and an exit code, rather than a traceback.
"""

from __future__ import annotations

import argparse
import sys

from logtriage.commands import gen_logs, profile, triage

COMMAND_MODULES = (gen_logs, profile, triage)


class LogtriageError(Exception):
    """An expected failure, reported as a styled error rather than a traceback.

    Anything raised that is *not* this is a bug in the tool, and a bug should
    print its traceback: swallowing every exception into a tidy message is how a
    tool ends up reporting "triage failed" for a typo in its own source.
    """


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m logtriage",
        description="Offline triage of mongod JSON logs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for module in COMMAND_MODULES:
        module.configure(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    from logtriage.output import Printer

    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except LogtriageError as exc:
        Printer(quiet=getattr(args, "json", False)).error(str(exc))
        return 1
    except BrokenPipeError:
        # `logtriage profile x.log | head` closes the pipe early. That is a
        # normal way to use a Unix tool, not a failure, and letting it surface
        # would print a traceback at the end of a perfectly good run.
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
