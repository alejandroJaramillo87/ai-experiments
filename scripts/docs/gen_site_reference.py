#!/usr/bin/env python3
"""Render site/docs/rust-harness/reference/<binary>.md from each binary's --help.

The harness renders --help from the same table its parser reads
(rust-harness/harness/src/cli.rs), and the seven runtime binaries do the same
from their Spec (rust-harness/runtime/src/cli.rs), so --help is the one
description of a command line that cannot drift from what is accepted. This
script turns that text into a page: the usage line in a fenced block, then one
markdown table per help section. scripts/docs/check_site_reference.py fails the
quality gate when a page and its table disagree.

The text is parsed rather than read from a machine-readable form because neither
crate has one, and adding one would mean widening the harness's private `Group`
table and touching eight binaries for a format three lines of rules cover. Both
renderers are pinned by golden tests, and their shape is regular: a line at
column zero ending in a colon opens a section, a two-space line is a term, a
six-space line continues the current term's description, and any other column
zero text is prose.

The binary set is discovered, not listed: `harness` plus every file under
runtime/src/bin/. A new binary gets a page on the next run without editing this
script; sidebars.js still needs a line, and the site build fails loudly on the
missing page until it has one.

Every binary answers --help before loading anything, so this needs no GPU, no
checkpoint, and no network beyond what cargo needs to build.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

REFERENCE_DIR = Path("site") / "docs" / "rust-harness" / "reference"

# The section the harness closes its help with. Its entries are binaries rather
# than flags, so the table gets different column names and each name links to
# that binary's own page.
BINARIES_SECTION = "Other binaries in this workspace"

# The indentation the two renderers use, as `help()` and `Spec::help()` write
# them: a term is indented two spaces, its description six.
TERM_INDENT = "  "
DESCRIPTION_INDENT = "      "


@dataclass
class Section:
    """One `Title:` block of help output and its term/description pairs."""

    title: str
    entries: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Help:
    """One binary's --help, parsed."""

    usage: str
    summary: list[str]
    sections: list[Section]
    trailing: list[str]


def parse_help(text: str) -> Help:
    """Split help text into its usage line, prose, and sections.

    Prose before the first section is the summary; prose after the last one is
    trailing. Consecutive prose lines join into one paragraph, a blank line
    separates paragraphs, and a wrapped description joins back into one string.
    """
    lines = text.splitlines()
    if not lines or not lines[0].startswith("usage: "):
        raise ValueError("help text does not begin with a usage line")
    usage = lines[0]

    summary: list[str] = []
    trailing: list[str] = []
    sections: list[Section] = []
    paragraph: list[str] = []

    def flush_prose() -> None:
        if paragraph:
            target = trailing if sections else summary
            target.append(" ".join(paragraph))
            paragraph.clear()

    for line in lines[1:]:
        if not line.strip():
            flush_prose()
            continue
        if line.startswith(DESCRIPTION_INDENT):
            if not sections or not sections[-1].entries:
                raise ValueError(f"description with no term: {line!r}")
            term, description = sections[-1].entries[-1]
            piece = line.strip()
            joined = f"{description} {piece}" if description else piece
            sections[-1].entries[-1] = (term, joined)
        elif line.startswith(TERM_INDENT):
            if not sections:
                raise ValueError(f"term outside any section: {line!r}")
            sections[-1].entries.append((line.strip(), ""))
        elif line.endswith(":"):
            flush_prose()
            sections.append(Section(line[:-1]))
        else:
            paragraph.append(line.strip())
    flush_prose()
    return Help(usage, summary, sections, trailing)


def cell(text: str) -> str:
    """Escape prose for a table cell: a pipe would split it, a bracket is HTML."""
    return text.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def code_cell(text: str) -> str:
    """Escape a backticked cell: inside a table a pipe splits even within code."""
    return "`" + text.replace("|", "\\|") + "`"


def render_page(name: str, source: str, help_text: Help, pages: set[str]) -> str:
    """The page for one binary. `pages` names the binaries that have a page."""
    out: list[str] = [
        "---",
        f"title: {name}",
        "---",
        "",
        f"# {name}",
        "",
        f"Generated by `make site-gen` from `{name} --help`. Do not edit this page; "
        f"edit the flag table in `{source}` and regenerate.",
        "",
    ]
    for paragraph in help_text.summary:
        out += [paragraph, ""]
    out += ["## Usage", "", "```text", help_text.usage, "```", ""]
    for section in help_text.sections:
        out += [f"## {section.title}", ""]
        if section.title == BINARIES_SECTION:
            out += ["| Binary | Purpose |", "|--------|---------|"]
            for term, description in section.entries:
                label = (
                    f"[{term}]({term}.md)" if term in pages and term != name else term
                )
                out.append(f"| {label} | {cell(description)} |")
        else:
            out += ["| Flag | Description |", "|------|-------------|"]
            for term, description in section.entries:
                out.append(f"| {code_cell(term)} | {cell(description)} |")
        out.append("")
    for paragraph in help_text.trailing:
        out += [paragraph, ""]
    return "\n".join(out)


def binaries(root: Path) -> list[tuple[str, str, str]]:
    """Each shipped binary as (name, cargo package, source path relative to root)."""
    found: list[tuple[str, str, str]] = []
    if (root / "rust-harness" / "harness" / "src" / "cli.rs").is_file():
        found.append(("harness", "harness", "rust-harness/harness/src/cli.rs"))
    bins_dir = root / "rust-harness" / "runtime" / "src" / "bin"
    for path in sorted(bins_dir.glob("*.rs")):
        found.append(
            (path.stem, "runtime", f"rust-harness/runtime/src/bin/{path.name}")
        )
    return found


def help_output(root: Path, package: str, name: str) -> str:
    """Run the binary with --help and return what it printed."""
    manifest = root / "rust-harness" / "Cargo.toml"
    result = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "--manifest-path",
            str(manifest),
            "-p",
            package,
            "--bin",
            name,
            "--",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    found = binaries(root)
    if not found:
        out.error("site-gen: enumerated no rust-harness binaries")
        out.line(
            f"  Expected rust-harness/harness/src/cli.rs or runtime/src/bin/*.rs under {root}"
        )
        return 1
    if not (root / "rust-harness" / "Cargo.toml").is_file():
        out.error(f"site-gen: no rust-harness workspace under {root}")
        return 1

    out_dir = root / REFERENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = {name for name, _, _ in found}
    for name, package, source in found:
        try:
            text = help_output(root, package, name)
        except subprocess.CalledProcessError as exc:
            out.error(f"site-gen: {name} --help exited {exc.returncode}")
            out.line(exc.stderr.rstrip())
            return 1
        page = out_dir / f"{name}.md"
        page.write_text(
            render_page(name, source, parse_help(text), pages), encoding="utf-8"
        )
        out.ok(f"{page.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
