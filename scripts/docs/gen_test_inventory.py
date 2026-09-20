#!/usr/bin/env python3
"""Render the inventory table in tests/<dir>/README.md from the test files themselves.

A directory of several dozen test modules is unreadable as a directory listing:
the file name says which module is under test and nothing about what it asserts.
The answer already exists, one per file, in the module docstring's first
sentence -- written by whoever wrote the tests, in the edit that added them. A
table copied out of those docstrings by hand is true on the day it is written
and wrong by the next merge, so this script copies them on demand instead.

Each row is a file, the number of `def test_` functions in it (module level and
inside classes, counted from the AST rather than by grepping, so a commented-out
test is not a test), and the first sentence of its module docstring. The table
is spliced between the marker comments in the directory's README, leaving the
hand-written prose above them alone.

--check regenerates in memory and diffs against what is committed, which is what
scripts/docs/check_test_inventory.py runs in the quality gate. A file with no
module docstring is an error naming the file: the docstring is the input, and a
row cannot be invented for a file that does not supply one.

The default directory is `training`, the only one with a generated inventory
today. --dir takes another so a second directory needs no new script.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

START_MARKER = "<!-- test-inventory:start -->"
END_MARKER = "<!-- test-inventory:end -->"

# A sentence ends at a period followed by whitespace or end of text. The
# lookahead is what keeps `training/models/head_loss.py:` and `0.5` from ending
# one, which is most of the periods in these docstrings.
SENTENCE_END = re.compile(r"\.(?=\s|$)")


def first_sentence(docstring: str) -> str:
    """The docstring's first sentence, whitespace-collapsed, with its period."""
    head = SENTENCE_END.split(docstring.strip(), maxsplit=1)[0]
    return " ".join(head.split()) + "."


def count_tests(tree: ast.Module) -> int:
    """`def test_` functions at module level and one level inside a class.

    A function nested inside another function is not collected by pytest, so it
    is not counted here either.
    """

    def is_test(node: ast.stmt) -> bool:
        return isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef
        ) and node.name.startswith("test_")

    total = sum(1 for node in tree.body if is_test(node))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            total += sum(1 for child in node.body if is_test(child))
    return total


def cell(text: str) -> str:
    """Escape prose for a table cell: an unescaped pipe would split the row."""
    return text.replace("|", "\\|")


def inventory(directory: Path) -> list[tuple[str, int, str]]:
    """Every test module in the directory as (file name, test count, subject).

    Raises ValueError naming the file when a module carries no docstring.
    """
    rows: list[tuple[str, int, str]] = []
    for path in sorted(directory.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree)
        if not docstring:
            raise ValueError(f"{path.name} has no module docstring")
        rows.append((path.name, count_tests(tree), first_sentence(docstring)))
    return rows


def render_table(rows: list[tuple[str, int, str]]) -> str:
    """The markdown table, without the marker comments around it."""
    lines = ["| File | Tests | Covers |", "|------|-------|--------|"]
    lines += [
        f"| `{name}` | {count} | {cell(covers)} |" for name, count, covers in rows
    ]
    return "\n".join(lines)


def split_readme(text: str, readme: Path) -> tuple[str, str, str]:
    """The README as (head, current table, tail), split on the marker comments.

    Raises ValueError naming the README when either marker is absent.
    """
    for marker in (START_MARKER, END_MARKER):
        if text.count(marker) != 1:
            raise ValueError(f"{readme} does not carry exactly one {marker}")
    head, rest = text.split(START_MARKER, 1)
    body, tail = rest.split(END_MARKER, 1)
    return head, body.strip("\n"), tail


def splice(text: str, table: str, readme: Path) -> str:
    """The README with `table` between the markers, replacing what was there."""
    head, _, tail = split_readme(text, readme)
    return f"{head}{START_MARKER}\n\n{table}\n\n{END_MARKER}{tail}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    parser.add_argument(
        "--dir",
        default="training",
        help="test directory under tests/ (default training)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and fail when the committed table differs",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    directory = root / "tests" / args.dir
    readme = directory / "README.md"
    label = f"tests/{args.dir}"
    if not directory.is_dir():
        out.error(f"test-inventory: no directory {label} under {root}")
        return 1
    if not readme.is_file():
        out.error(f"test-inventory: no README.md in {label}")
        out.line("  The generated table needs a README to be spliced into.")
        return 1

    try:
        rows = inventory(directory)
    except ValueError as exc:
        out.error(f"test-inventory: {exc}")
        out.line("  The first sentence of the module docstring is the Covers column.")
        out.line("  Solution: add a docstring whose first sentence names the subject")
        return 1

    # The empty-set guard. A directory with no test modules would render a
    # header and no rows, and --check would then hold the README to it.
    if not rows:
        out.error(f"test-inventory: {label} carries no test_*.py modules")
        return 1

    table = render_table(rows)
    text = readme.read_text(encoding="utf-8")
    try:
        _, committed, _ = split_readme(text, readme.relative_to(root))
    except ValueError as exc:
        out.error(f"test-inventory: {exc}")
        out.line(f"  Solution: add the markers around the table in {label}/README.md")
        out.line(f"  Markers: {START_MARKER} and {END_MARKER}")
        return 1

    if args.check:
        if committed == table:
            out.ok(f"test-inventory: {label}/README.md lists {len(rows)} files")
            return 0
        out.error(f"test-inventory: {label}/README.md is stale")
        diff = difflib.unified_diff(
            committed.splitlines(),
            table.splitlines(),
            fromfile="committed",
            tofile="generated",
            lineterm="",
        )
        for line in diff:
            out.line(f"  {line}")
        out.line("  Solution: regenerate the table")
        out.line(
            f"  Command: python scripts/docs/gen_test_inventory.py --dir {args.dir}"
        )
        return 1

    readme.write_text(splice(text, table, readme.relative_to(root)), encoding="utf-8")
    out.ok(f"{readme.relative_to(root)}: {len(rows)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
