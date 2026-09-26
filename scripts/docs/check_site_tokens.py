#!/usr/bin/env python3
"""Fail the quality gate when a design token the site mirrors from figura differs from figura's value.

site/src/css/custom.css declares, under the blog's names, the panel surfaces,
shape, motion and type that the site shares with the blog and with figura, so
code blocks read as the same material as the diagrams. figura's
shared/tokens.css is the source of those values. The stylesheet said the two
were identical and nothing compared them, and --font-mono was not.

The comparison is the blog's own, tests/check-figura-tokens.mjs in
curiosity-chronicles: an explicit pair table from a site name to a figura name,
values compared after collapsing whitespace and case, every paired token
declared once, in the first :root block. The figura side is not read from a
checkout. It is the snapshot of shared/tokens.css pinned in
site/static/img/diagrams/figura.json at the same commit the diagram copies came
from, so the gate runs offline and a palette bump in figura reaches the site
only as a deliberate update of that record.

A figura token the site knowingly declares with a different value is listed in
UNPAIRED rather than left out of the table, so the difference stays visible in
the gate's output instead of passing as an oversight.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

STYLESHEET = Path("site") / "src" / "css" / "custom.css"
RECORD = Path("site") / "static" / "img" / "diagrams" / "figura.json"

# Site token -> figura token. The same ten values the blog pairs for its code
# panels, under the same names. Site-only tokens (--panel-shadow, --radius-sm,
# --dur, the page neutrals and the page accent) have no figura counterpart and
# are not paired.
PAIRS: list[tuple[str, str]] = [
    ("--panel-bg", "--fg-bg"),
    ("--panel-2", "--fg-panel"),
    ("--panel-fg", "--fg-text"),
    ("--panel-muted", "--fg-muted"),
    ("--panel-faint", "--fg-line"),
    ("--panel-border", "--fg-border"),
    ("--panel-accent", "--fg-accent"),
    ("--radius", "--fg-radius"),
    ("--ease", "--fg-ease"),
    ("--font-sans", "--fg-font"),
]

# Site token -> the figura token it corresponds to but is not compared with.
# --font-mono prefers JetBrains Mono, Fira Code and Cascadia Code ahead of the
# platform monospace; figura's --fg-mono names SFMono-Regular instead. The blog
# pairs its --font-mono with --fg-mono. Whether the site converges on figura's
# stack is an open decision for the author, so neither value changes here and
# the pair moves into PAIRS when it is made.
UNPAIRED: dict[str, str] = {
    "--font-mono": "--fg-mono",
}

ROOT_BLOCK = re.compile(r":root\s*\{([\s\S]*?)\n\}")
DECLARATION = re.compile(r"(--[a-z0-9-]+)\s*:\s*([^;]+);")
COMMENT = re.compile(r"/\*[\s\S]*?\*/")


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def blank_comments(css: str) -> str:
    """The stylesheet with each comment replaced by spaces, keeping offsets and lines."""
    return COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), css)


def root_block(css: str) -> re.Match[str] | None:
    """The first :root block of a comment-blanked stylesheet."""
    return ROOT_BLOCK.search(blank_comments(css))


def declarations(block: str) -> dict[str, str]:
    """Custom property name -> normalized value, for one block's body."""
    return {m.group(1): normalize(m.group(2)) for m in DECLARATION.finditer(block)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    try:
        record = json.loads((root / RECORD).read_text(encoding="utf-8"))
        css = (root / STYLESHEET).read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        out.error(f"site-tokens: cannot read the inputs: {exc}")
        return 1

    tokens = record.get("tokens") if isinstance(record, dict) else None
    snapshot = tokens.get("values") if isinstance(tokens, dict) else None
    # The empty-set guard. An empty snapshot would fail every pair rather than
    # none, which is a confusing way to learn the record lost its tokens.
    if not isinstance(snapshot, dict) or not snapshot:
        out.error(f"site-tokens: {RECORD} carries no token snapshot")
        out.line("  Expected `tokens.values`: figura token -> value, from")
        out.line("  shared/tokens.css at the record's figura commit.")
        return 1
    figura = {name: normalize(str(value)) for name, value in snapshot.items()}

    blanked = blank_comments(css)
    match = ROOT_BLOCK.search(blanked)
    if match is None:
        out.error(f"site-tokens: {STYLESHEET} has no :root block")
        return 1
    site = declarations(match.group(1))
    rest = (
        blanked[: match.start()]
        + "\n" * match.group(0).count("\n")
        + blanked[match.end() :]
    )

    findings: list[str] = []
    for site_name, figura_name in PAIRS:
        if site_name not in site:
            findings.append(
                f"{STYLESHEET}: {site_name} missing from the first :root block"
            )
        elif figura_name not in figura:
            findings.append(
                f"{RECORD}: {figura_name} missing from the snapshot "
                f"(paired with {site_name})"
            )
        elif site[site_name] != figura[figura_name]:
            findings.append(
                f"{STYLESHEET}: {site_name} is {site[site_name]}, figura "
                f"{figura_name} is {figura[figura_name]}"
            )
        redeclared = re.compile(rf"^\s*{re.escape(site_name)}\s*:", re.MULTILINE)
        for other in redeclared.finditer(rest):
            line = rest.count("\n", 0, other.start()) + 1
            findings.append(
                f"{STYLESHEET}:{line}: {site_name} is redeclared outside the first "
                ":root block; a mirrored token is declared once"
            )

    for site_name in UNPAIRED:
        if site_name not in site:
            findings.append(
                f"{STYLESHEET}: {site_name} is listed unpaired but not declared "
                "in the first :root block"
            )

    named = {figura_name for _, figura_name in PAIRS} | set(UNPAIRED.values())
    for figura_name in sorted(set(figura) - named):
        findings.append(
            f"{RECORD}: {figura_name} is in the snapshot but paired with nothing"
        )

    unpaired = ", ".join(f"{site} ({fig})" for site, fig in UNPAIRED.items()) or "none"
    out.info(f"site-tokens: {len(PAIRS)} paired, {len(UNPAIRED)} unpaired: {unpaired}")
    if not findings:
        out.ok("site-tokens: every paired token equals figura's pinned value")
        return 0

    noun = "finding" if len(findings) == 1 else "findings"
    out.error(f"site-tokens: {len(findings)} {noun}")
    for finding in findings:
        out.line(f"  {finding}")
    out.line(
        "  Solution: figura's shared/tokens.css is the source. Change the value in"
    )
    out.line(
        "  figura first, update the snapshot in the record at the new commit, then"
    )
    out.line("  mirror it in custom.css; a deliberate difference belongs in UNPAIRED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
