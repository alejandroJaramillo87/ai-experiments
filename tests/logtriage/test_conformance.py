"""Hold the copied logtriage Printer in step with the canonical one.

logtriage/ is a self-contained project: it imports nothing from this repository
so that it can be lifted out whole onto another machine. The cost of that is one
deliberate copy, logtriage/logtriage/output.py, of common/output.py --
.styles/PYTHON_OUTPUT_STYLE.md says to import the canonical Printer rather than
copy it, and this is the documented exception.

A copy nobody checks drifts. This module is the check: it drives both Printers
through the same call sequence and fails if a single byte of stderr differs. It
does not travel on extraction, which is the intent -- it exists to keep the two
in step for as long as they are neighbours, and the extracted project has no
second Printer to disagree with.

Deliberately NOT compared: the structured file record. common/output.py emits one
through common/obs on every status call and the copy does not, because common/obs
is this repository's infrastructure (its own log root, its own run ids). The
contract that has to hold across the two is what a human or a pipe sees.

The module is loaded from an explicit path rather than by `import logtriage`.
The repository root holds a directory named logtriage/, so a bare import there
resolves to a namespace package covering the project directory rather than the
package inside it, and which one wins depends on sys.path order -- the same
environment-dependent shadowing that pyproject.toml's known-first-party comment
was written about.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from common.output import Printer as CanonicalPrinter

REPO_ROOT = Path(__file__).resolve().parents[2]
COPY_PATH = REPO_ROOT / "logtriage" / "logtriage" / "output.py"


def _load_copy() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_logtriage_output", COPY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Every public method, each status tag, both indent behaviours, and the blank
# line -- the sequence is the contract, so a method added to one Printer and not
# the other shows up as an AttributeError here rather than as a surprise on the
# Mac.
CALLS = [
    ("line", ("=== logtriage ===",)),
    ("line", ()),
    ("ok", ("Parsed records: 1200",)),
    ("ok", ("Slow queries: 14", 2)),
    ("warn", ("Truncated lines: 3",)),
    ("error", ("Store unreachable: 127.0.0.1:27017",)),
    ("info", ("Window: 15m",)),
    ("run", ("Templating",)),
    ("skip", ("Vector index: no mongot", 4)),
    ("line", ("  Solution: start the stack",)),
]


def _drive(printer: object) -> None:
    for name, args in CALLS:
        getattr(printer, name)(*args)


@pytest.mark.parametrize("quiet", [False, True])
def test_printer_stderr_is_byte_identical(capsys, monkeypatch, quiet: bool) -> None:
    # NO_COLOR set: pytest's captured stderr is not a TTY either, so this pins
    # the plain-text branch by two independent conditions rather than relying on
    # capsys never being a TTY.
    monkeypatch.setenv("NO_COLOR", "1")

    _drive(CanonicalPrinter(quiet=quiet))
    canonical = capsys.readouterr()

    _drive(_load_copy().Printer(quiet=quiet))
    copied = capsys.readouterr()

    assert copied.err == canonical.err
    # stdout stays reserved for --json in both; a status line leaking there
    # would corrupt machine output.
    assert canonical.out == ""
    assert copied.out == ""


def test_color_codes_are_identical() -> None:
    """The escape sequences themselves, which the plain-text runs cannot see.

    capsys is never a TTY, so the comparison above exercises only the uncolored
    branch. Comparing the tables directly covers the branch a real terminal takes.
    """
    canonical = CanonicalPrinter(quiet=False)
    copied = _load_copy().Printer(quiet=False)
    # Constructed off-TTY, so both tables are empty; force the colored branch by
    # comparing the literal tables the colored branch would install.
    canonical_src = (REPO_ROOT / "common" / "output.py").read_text()
    copied_src = COPY_PATH.read_text()
    for tag in ("OK", "WARN", "ERROR", "INFO", "RUN", "SKIP"):
        marker = f'"{tag}": "\\033['
        assert marker in canonical_src, f"canonical lost the {tag} color"
        assert marker in copied_src, f"copy lost the {tag} color"
        start = canonical_src.index(marker)
        end = canonical_src.index('"', start + len(marker))
        canonical_code = canonical_src[start:end]
        start = copied_src.index(marker)
        end = copied_src.index('"', start + len(marker))
        assert copied_src[start:end] == canonical_code
    assert canonical._c == copied._c == {}


def test_copy_imports_nothing_first_party() -> None:
    """The property that makes extraction work, asserted rather than assumed.

    An `from common.obs import core` added here for convenience would pass every
    test above and break the moment the directory is copied to another machine.
    """
    source = COPY_PATH.read_text()
    for forbidden in ("common.", "from common", "import common", "interp.", "training."):
        assert forbidden not in source, f"logtriage/output.py reaches for {forbidden}"


def test_copy_is_loadable_without_repo_on_path() -> None:
    """Loading the copy must not depend on the parent repository being importable."""
    saved = sys.path[:]
    try:
        sys.path[:] = [p for p in sys.path if Path(p).resolve() != REPO_ROOT]
        module = _load_copy()
        assert hasattr(module, "Printer")
    finally:
        sys.path[:] = saved
