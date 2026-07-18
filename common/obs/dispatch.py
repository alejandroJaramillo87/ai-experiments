"""Traced CLI dispatch shared by the training and interp mains (roadmap OBS-4).

Both CLIs share the dispatch shape ``args = build_parser().parse_args(argv);
return args.func(args)``. dispatch() wraps it: parse, generate a run id,
configure the logging core, and bracket the handler between cmd.start and
cmd.end events so every invocation is reconstructable from its event file.
Argparse usage errors (SystemExit(2)) escape during parsing, before a run id
exists, and produce no records.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence

from common.obs.audit import install_audit_trace
from common.obs.core import bind_run, configure, emit, record_failure


def dispatch(
    cli: str,
    build_parser: Callable[[], argparse.ArgumentParser],
    argv: Sequence[str] | None = None,
) -> int:
    """Parse, run, and record one CLI invocation; return its exit code.

    Exceptions re-raise after recording, so the interpreter still prints
    the traceback and the process exit status is unchanged. cmd.end always
    fires, recording an uncaught exception as exit code 1, KeyboardInterrupt
    as 130, and a mid-run SystemExit (the SIGTERM traps) as its own code.
    """
    args = build_parser().parse_args(argv if argv is None else list(argv))
    # OBS_TRACE=1 implies debug level (an explicit OBS_LOG_LEVEL wins);
    # without this the debug audit records would be dropped unwritten.
    trace = os.environ.get("OBS_TRACE") == "1"
    level = "debug" if trace and not os.environ.get("OBS_LOG_LEVEL") else None
    run_id = configure(cli, level=level)
    bind_run(cli, args.command, run_id)
    if trace:
        install_audit_trace()
    emit(
        "info",
        "",
        event="cmd.start",
        console=False,
        argv=list(argv) if argv is not None else sys.argv[1:],
    )
    start = time.monotonic()
    exit_code = 1
    try:
        result = args.func(args)
        exit_code = 0 if result is None else int(result)
        return exit_code
    except KeyboardInterrupt:
        exit_code = 130
        raise
    except SystemExit as exc:
        if isinstance(exc.code, int):
            exit_code = exc.code
        else:
            exit_code = 0 if exc.code is None else 1
        raise
    except BaseException as exc:
        # KeyboardInterrupt and SystemExit are handled above: cancellation
        # and a deliberate exit are not failures, so only genuine uncaught
        # exceptions produce a fail record.
        exit_code = 1
        record_failure(exc, exit_code)
        raise
    finally:
        emit(
            "info",
            "",
            event="cmd.end",
            console=False,
            exit_code=exit_code,
            duration_s=round(time.monotonic() - start, 3),
        )
