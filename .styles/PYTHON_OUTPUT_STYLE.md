# Python Output Style Guide

Output formatting for Python CLI utilities (e.g. `scripts/utils/*.py`). Follow Unix
conventions: parseable, minimal, consistent, no decoration or emojis, clear about
required user actions. Exit codes are the primary success signal (0 = success);
human-readable lines are secondary and must stay separable from machine output.

Consistent with `.styles/BASH_OUTPUT_STYLE.md` — the same status vocabulary and
`[STATUS] Subject: detail` format, so bash and Python tools read identically.

## Status Indicators

Bracketed format with ANSI colors:

| Indicator | Color  | Usage |
|-----------|--------|-------|
| `[OK]`    | Green  | Operation completed successfully |
| `[WARN]`  | Yellow | Non-critical issue requiring attention |
| `[ERROR]` | Red    | Critical failure requiring action |
| `[INFO]`  | Cyan   | Informational message |
| `[RUN]`   | Blue   | Operation in progress |
| `[SKIP]`  | Blue   | Operation skipped |

Format is `[STATUS] Subject: detail`, e.g. `[OK] PyTorch version: 2.9.1+cu130`.

## Color Setup

Use standard ANSI codes. Disable color when **any** of: `NO_COLOR` is set, stdout is
not a TTY (piped/redirected), or the tool is in JSON/quiet mode. Detecting the TTY
keeps escape codes out of pipes and log files without needing a flag.

A canonical reusable helper — a small `Printer` class exposing status methods:

```python
import os
import sys


class Printer:
    """Styled status output. Colorized for a TTY, plain otherwise.

    In quiet mode (e.g. --json) all human output is suppressed so stdout stays
    machine-parseable.
    """

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        use_color = (
            not quiet
            and not os.environ.get("NO_COLOR")
            and sys.stdout.isatty()
        )
        if use_color:
            self._c = {
                "OK": "\033[0;32m", "WARN": "\033[1;33m", "ERROR": "\033[0;31m",
                "INFO": "\033[0;36m", "RUN": "\033[0;34m", "SKIP": "\033[0;34m",
            }
            self._nc = "\033[0m"
        else:
            self._c = {}
            self._nc = ""

    def _status(self, tag: str, message: str, indent: int = 0) -> None:
        if self.quiet:
            return
        color = self._c.get(tag, "")
        pad = " " * indent
        print(f"{pad}{color}[{tag}]{self._nc} {message}")

    def ok(self, m, indent=0):    self._status("OK", m, indent)
    def warn(self, m, indent=0):  self._status("WARN", m, indent)
    def error(self, m, indent=0): self._status("ERROR", m, indent)
    def info(self, m, indent=0):  self._status("INFO", m, indent)
    def skip(self, m, indent=0):  self._status("SKIP", m, indent)

    def line(self, message: str = "") -> None:
        """Plain line (headers, blank lines) — still suppressed when quiet."""
        if not self.quiet:
            print(message)
```

Always reset color with the reset code. Colors are display-only; output must stay
parseable with `NO_COLOR` or when piped.

## Structure

Simple text headers, 2-space indentation per hierarchy level:

```python
out.line("=== PyTorch GPU validation ===")
out.line()
out.ok("CUDA availability: True")
out.ok("Blackwell sm_120: present in torch arch list", indent=2)
```

Renders as:

```
=== PyTorch GPU validation ===

[OK] CUDA availability: True
  [OK] Blackwell sm_120: present in torch arch list
```

Spacing: 1 blank line after major sections, no blank lines between related items.

## Errors and Warnings

Include context and a solution:

```python
out.error("Model file not found: /mnt/ai-data/models/x")
out.line("  Solution: Verify the model exists at the specified path")
out.line("  Command: ls -la /mnt/ai-data/models")

out.warn("bitsandbytes: 0.46.1 (CUDA backend unavailable)")
out.line("  Solution: reinstall bitsandbytes matching the installed CUDA (cu130)")
```

## JSON / Machine Mode

When `--json` (or another machine-output flag) is set, stdout must contain **only**
valid JSON. Construct the `Printer` with `quiet=True` so every human line is
suppressed, then emit one `json.dumps(...)` at the end. Never interleave prose with
the JSON payload.

```python
out = Printer(quiet=args.json)
...
if args.json:
    print(json.dumps(result, indent=2))
```

## Suppressing Third-Party Import Noise

Heavy AI imports emit warnings and, worse, print tracebacks to stderr at import time
even when the import "succeeds" (e.g. `bitsandbytes` when its CUDA binary is missing,
or a pynvml `FutureWarning` from `torch`). Never let a dependency's noise bleed into
the tool's structured output. Filter warnings and capture stderr/stdout around the
import, then decide what to surface yourself:

```python
import contextlib
import io
import warnings


def quiet_import(module_name: str):
    """Import a module, capturing its import-time stderr/warnings.

    Returns (module_or_None, captured_stderr). Inspect the captured text to
    surface real problems as a clean [WARN], instead of dumping a traceback.
    """
    buf = io.StringIO()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(buf):
            warnings.simplefilter("ignore")
            module = __import__(module_name)
        return module, buf.getvalue()
    except Exception:
        return None, buf.getvalue()
```

Then translate a known failure signature into a single styled line:

```python
mod, noise = quiet_import("bitsandbytes")
if mod and "Configured CUDA binary not found" in noise:
    out.warn("bitsandbytes: CUDA backend unavailable")
    out.line("  Solution: reinstall bitsandbytes matching the installed CUDA")
```

## Exit Codes

- `0` — success
- `1` — validation/runtime failure
- `2` — invalid usage (bad arguments)

Return the code from `main()` and `sys.exit(main())`; do not call `sys.exit` with a
message string in normal flow — print the styled error, then return the code.

## What to Avoid

- Decorative borders or emphasis banners (`# ==== SUPER IMPORTANT ====`), emojis, or
  Unicode ornamentation.
- Printing raw tracebacks in normal output — capture and translate to `[WARN]`/
  `[ERROR]` with a solution.
- Emitting ANSI color when stdout is not a TTY or `NO_COLOR` is set.
- Interleaving human prose with JSON in machine mode.
- Vague language ("blazingly fast") — state the technical fact.

## Related Guides

- `.styles/BASH_OUTPUT_STYLE.md`   — bash script output formatting
- `.styles/TEST_OUTPUT_STYLE.md`   — test program/runner output format
- `.styles/COMMENT_STYLE.md`       — comment conventions (all languages)
