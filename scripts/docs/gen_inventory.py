#!/usr/bin/env python3
"""Render the inventory tables in a README from the tree the README describes.

A directory of several dozen files is unreadable as a directory listing: the
file name says what a module is about and nothing about what it does. The
answer already exists, once per file, in a place its author wrote it in the
same edit: the first sentence of a module docstring, or the description
paragraph under a README's title, which the documentation style guide requires.
A table copied out of those by hand is true on the day it is written and wrong
by the next merge, so this script copies them on demand instead.

A README declares what it wants with a marker pair, and the kind is carried in
the marker so no table of READMEs has to be kept in step:

    <!-- inventory:tests:start -->    one row per test_*.py in the README's
                                      directory: file, `def test_` count, the
                                      docstring's first sentence
    <!-- inventory:modules:start -->  one row per *.py except __init__ and
                                      __main__: module, the docstring's first
                                      sentence
    <!-- inventory:dirs:start -->     one row per child directory: a link to
                                      its README and that README's first
                                      sentence
    <!-- inventory:pages:start -->    one row per sibling *.md except the
                                      README: a link and its first sentence
    <!-- inventory:scripts:start -->  one row per *.sh, *.py and *.mjs: the
                                      script and the first sentence of its
                                      header comment or docstring
    <!-- inventory:contracts:start --> one row per *.json contract: its
                                      description's first sentence and the
                                      direction its writers and readers imply
    <!-- inventory:roles:start -->    one row per .claude/agents/*.md: tools
                                      and the frontmatter description
    <!-- inventory:commands:start --> one row per .claude/commands/*.md
    <!-- inventory:skills:start -->   one row per .claude/skills/*/SKILL.md
    <!-- inventory:fragments:start --> one row per *.mk: its `##@` section
                                      title and the targets `make help` lists

The first six and the last read the README's own directory. The three roster kinds read a
fixed place under the repository root, because the definitions they list live
in .claude/ and the page that indexes them lives in docs/agents/; the source is
fixed by the kind the way the hub roots are fixed in check_inventory.py.

Each pair closes with the matching `<!-- inventory:<kind>:end -->`, and a README
may carry more than one pair. The table is spliced between the markers, leaving
the hand-written prose around them alone. A file with no docstring, a page with
no description paragraph, or a child directory with no README is an error
naming it: the source is the input, and a row cannot be invented for a source
that does not supply one. That last error is what makes a hub README's table
the rule that every directory under it is documented.

With no README argument the script finds every README in the tree that carries
a marker pair. --check regenerates in memory and diffs against what is
committed, which is what scripts/docs/check_inventory.py runs in the quality
gate.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

KINDS = (
    "tests",
    "modules",
    "dirs",
    "pages",
    "scripts",
    "contracts",
    "roles",
    "commands",
    "skills",
    "fragments",
)
MARKER = re.compile(r"<!-- inventory:(\w+):(start|end) -->")

# Directories a walk never enters, the same set the sibling gates prune.
PRUNE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "__pycache__",
    ".ruff_cache",
    ".pytest-tmp",
}

# The files a scripts table lists, and how each carries its purpose: a docstring
# for Python, a leading `#` block for shell, a leading `/* */` block for a module.
SCRIPT_SUFFIXES = (".sh", ".py", ".mjs")

# Where each roster kind reads its definitions, relative to the repository root.
ROSTER_SOURCES = {
    "roles": (".claude/agents", "*.md"),
    "commands": (".claude/commands", "*.md"),
    "skills": (".claude/skills", "*/SKILL.md"),
}

# The language a contract's writer or reader path implies. Anything else, such
# as an upstream config.json or a transcript under docs/, is upstream.
LANGUAGES = (
    (re.compile(r"rust-harness\b"), "Rust"),
    (re.compile(r"(interp|training|common|tests)\b"), "Python"),
)
LANGUAGE_ORDER = ("Rust", "Python", "Upstream")

# A make fragment's section title, and a rule `make help` lists: the awk in
# make/help.mk prints a rule when its recipe line carries `##`, so the table
# reads the same two comment shapes and nothing else.
SECTION_LINE = re.compile(r"^##@\s*(.+?)\s*$")
HELP_RULE = re.compile(r"^([a-zA-Z0-9_][a-zA-Z0-9_-]*)\s*:.*?##")

# A header line that is only a filename, or opens with one and a separator: the
# older scripts name themselves before saying what they do, and a few name a
# file they were once called.
FILENAME_LINE = re.compile(r"[\w.-]+\.(sh|py|mjs)\s*(--?|—|:)?\s*")

# A sentence ends at a period followed by whitespace or end of text. The
# lookahead is what keeps `training/models/head_loss.py:` and `0.5` from ending
# one, which is most of the periods in these docstrings.
SENTENCE_END = re.compile(r"\.(?=\s|$)")

Row = tuple[str, ...]


def first_sentence(text: str) -> str:
    """The text's first sentence, whitespace-collapsed, with its period."""
    head = SENTENCE_END.split(text.strip(), maxsplit=1)[0]
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


def description(page: Path, root: Path) -> str:
    """The first sentence of the description paragraph under a page's title.

    The style guide puts one H1 first, then an optional bold class line for a
    frozen record, an operational runbook or a parked proposal, then the
    paragraph that says what the document is. That paragraph is the source; a
    heading, a table, a list or a fenced block in its place means the page has
    no description, and that is an error rather than a guess.
    """
    lines = page.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines) and not lines[index].startswith("# "):
        index += 1
    if index == len(lines):
        raise ValueError(f"{page.relative_to(root)} has no H1 title")
    index += 1
    paragraph: list[str] = []
    in_class_line = False
    for line in lines[index:]:
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            in_class_line = False
            continue
        if not paragraph and not in_class_line and stripped.startswith("**"):
            # The class line: bold, one paragraph, never the description.
            in_class_line = True
            continue
        if in_class_line:
            continue
        if stripped.startswith(("```", "#", "|", "- ", "> ")) or re.match(
            r"\d+\. ", stripped
        ):
            break
        paragraph.append(stripped)
    if not paragraph:
        raise ValueError(
            f"{page.relative_to(root)} has no description paragraph under its title"
        )
    return first_sentence(" ".join(paragraph))


def module_sentence(path: Path, root: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree)
    if not docstring:
        raise ValueError(f"{path.relative_to(root)} has no module docstring")
    return first_sentence(docstring)


def header_paragraph(comments: list[str], path: Path, root: Path) -> str:
    """The first sentence of a script's leading comment block.

    `comments` is the block's text with the comment syntax removed, one entry per
    line, where a bare comment line is an empty string. The paragraph is the
    lines up to the first blank one; a line that is only a filename is dropped
    and a line opening with a filename and a separator loses that prefix, so
    `guard.sh - single-source guardrail` and a stale `old_name.sh` line above
    the description both yield the description.
    """
    paragraph: list[str] = []
    for text in comments:
        if not text.strip():
            if paragraph:
                break
            continue
        text = text.strip()
        match = FILENAME_LINE.match(text)
        if match and match.end() == len(text):
            continue
        if match:
            text = text[match.end() :]
        paragraph.append(text)
    if not paragraph:
        raise ValueError(f"{path.relative_to(root)} has no header comment")
    sentence = first_sentence(" ".join(paragraph))
    # A description that followed `name.sh -` on its line starts lowercase.
    return sentence[0].upper() + sentence[1:]


def shell_comments(lines: list[str]) -> list[str]:
    """The first `#` block of a shell script, after the shebang and any `set`."""
    block: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#!"):
            continue
        if stripped.startswith("#"):
            block.append(stripped[1:])
        elif block or (stripped and not stripped.startswith("set ")):
            break
    return block


def module_comments(lines: list[str]) -> list[str]:
    """The text of a JavaScript module's leading `/* */` block."""
    block: list[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if not inside:
            if stripped.startswith("/*"):
                inside = True
                stripped = stripped[2:].strip()
                if stripped:
                    block.append(stripped)
            elif stripped:
                break
            continue
        if "*/" in stripped:
            break
        block.append(stripped[1:] if stripped.startswith("*") else stripped)
    return block


def script_sentence(path: Path, root: Path) -> str:
    if path.suffix == ".py":
        return module_sentence(path, root)
    lines = path.read_text(encoding="utf-8").splitlines()
    comments = shell_comments(lines) if path.suffix == ".sh" else module_comments(lines)
    return header_paragraph(comments, path, root)


def rows_scripts(directory: Path, root: Path) -> list[Row]:
    return [
        (path.name, script_sentence(path, root))
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in SCRIPT_SUFFIXES
    ]


def language(entry: str) -> str:
    for pattern, name in LANGUAGES:
        if pattern.match(entry):
            return name
    return "Upstream"


def who(entries: list[str], verb: str) -> str:
    """`Rust and Python write`, or `Python reads`: the languages, then the verb."""
    found = {language(entry) for entry in entries}
    names = [name for name in LANGUAGE_ORDER if name in found]
    if len(names) == 1:
        return f"{names[0]} {verb}s"
    return f"{', '.join(names[:-1])} and {names[-1]} {verb}"


def direction(contract: dict[str, object]) -> str:
    """`Rust writes, Python reads`, derived from the writers and readers lists."""
    writers = contract.get("writers") or []
    readers = contract.get("readers") or []
    if not isinstance(writers, list) or not writers:
        raise ValueError("no writers")
    if not isinstance(readers, list) or not readers:
        return f"{who(writers, 'write')}, no reader yet"
    return f"{who(writers, 'write')}, {who(readers, 'read')}"


def rows_contracts(directory: Path, root: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(directory.glob("*.json")):
        contract = json.loads(path.read_text(encoding="utf-8"))
        text = contract.get("description")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path.relative_to(root)} has no description")
        try:
            heading = direction(contract)
        except ValueError as exc:
            raise ValueError(f"{path.relative_to(root)} has {exc}") from None
        rows.append((path.name, first_sentence(text), heading))
    return rows


def frontmatter(path: Path) -> dict[str, str]:
    """The `key: value` lines between a markdown file's opening `---` fences."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and key.strip():
            fields[key.strip()] = value.strip()
    return fields


def rows_roster(kind: str, root: Path) -> list[Row]:
    source, pattern = ROSTER_SOURCES[kind]
    rows: list[Row] = []
    for path in sorted((root / source).glob(pattern)):
        fields = frontmatter(path)
        text = fields.get("description", "")
        if not text:
            raise ValueError(f"{path.relative_to(root)} has no description")
        name = path.parent.name if kind == "skills" else path.stem
        if kind == "roles":
            rows.append((name, fields.get("tools", ""), first_sentence(text)))
        else:
            rows.append((name, first_sentence(text)))
    return rows


def fragment_row(path: Path, root: Path) -> Row:
    """The section title and help-listed targets of one make fragment."""
    section = None
    targets: list[str] = []
    for line in path.read_text().splitlines():
        if section is None:
            if not line.strip():
                continue
            match = SECTION_LINE.match(line)
            if match is None:
                raise ValueError(f"{path.relative_to(root)}: no ##@ section line")
            section = match.group(1)
            continue
        match = HELP_RULE.match(line)
        if match is not None:
            targets.append(match.group(1))
    if section is None:
        raise ValueError(f"{path.relative_to(root)}: no ##@ section line")
    return (path.name, section, ", ".join(f"`{name}`" for name in targets))


def rows_fragments(directory: Path, root: Path) -> list[Row]:
    return [fragment_row(path, root) for path in sorted(directory.glob("*.mk"))]


def rows_tests(directory: Path, root: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(directory.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree)
        if not docstring:
            raise ValueError(f"{path.relative_to(root)} has no module docstring")
        rows.append((path.name, str(count_tests(tree)), first_sentence(docstring)))
    return rows


def rows_modules(directory: Path, root: Path) -> list[Row]:
    return [
        (path.name, module_sentence(path, root))
        for path in sorted(directory.glob("*.py"))
        if path.name not in ("__init__.py", "__main__.py")
    ]


def child_dirs(directory: Path) -> list[Path]:
    """Child directories a hub table lists: not pruned, not hidden, not empty once pruned."""
    children = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or child.name in PRUNE_DIRS:
            continue
        if child.name.startswith("."):
            # Tool state and configuration, not a place a reader is sent.
            continue
        if not any(entry.name not in PRUNE_DIRS for entry in child.iterdir()):
            continue
        children.append(child)
    return children


def rows_dirs(directory: Path, root: Path) -> list[Row]:
    rows: list[Row] = []
    for child in child_dirs(directory):
        readme = child / "README.md"
        if not readme.is_file():
            raise ValueError(f"{child.relative_to(root)}/ has no README.md")
        rows.append((child.name, description(readme, root)))
    return rows


def rows_pages(directory: Path, root: Path) -> list[Row]:
    return [
        (page.name, description(page, root))
        for page in sorted(directory.glob("*.md"))
        if page.name != "README.md"
    ]


ROWS = {
    "tests": rows_tests,
    "modules": rows_modules,
    "dirs": rows_dirs,
    "pages": rows_pages,
    "scripts": rows_scripts,
    "contracts": rows_contracts,
    "roles": lambda directory, root: rows_roster("roles", root),
    "commands": lambda directory, root: rows_roster("commands", root),
    "skills": lambda directory, root: rows_roster("skills", root),
    "fragments": rows_fragments,
}


def render_table(kind: str, rows: list[Row]) -> str:
    """The markdown table for one kind, without the marker comments around it."""
    if kind == "tests":
        lines = ["| File | Tests | Covers |", "|------|-------|--------|"]
        lines += [f"| `{n}` | {c} | {cell(s)} |" for n, c, s in rows]
    elif kind == "modules":
        lines = ["| Module | Purpose |", "|--------|---------|"]
        lines += [f"| `{n}` | {cell(s)} |" for n, s in rows]
    elif kind == "dirs":
        lines = ["| Directory | Holds |", "|-----------|-------|"]
        lines += [f"| [{n}/]({n}/README.md) | {cell(s)} |" for n, s in rows]
    elif kind == "scripts":
        lines = ["| Script | Purpose |", "|--------|---------|"]
        lines += [f"| `{n}` | {cell(s)} |" for n, s in rows]
    elif kind == "contracts":
        lines = ["| Contract | Seam | Direction |", "|----------|------|-----------|"]
        lines += [f"| [{n}]({n}) | {cell(s)} | {d} |" for n, s, d in rows]
    elif kind == "roles":
        lines = ["| Role | Tools | Does |", "|------|-------|------|"]
        lines += [f"| `{n}` | {t} | {cell(s)} |" for n, t, s in rows]
    elif kind == "commands":
        lines = ["| Command | Does |", "|---------|------|"]
        lines += [f"| `/{n}` | {cell(s)} |" for n, s in rows]
    elif kind == "skills":
        lines = ["| Skill | Does |", "|-------|------|"]
        lines += [f"| `{n}` | {cell(s)} |" for n, s in rows]
    elif kind == "fragments":
        lines = ["| Fragment | Section | Targets |", "|----------|---------|---------|"]
        lines += [f"| `{n}` | {cell(s)} | {t} |" for n, s, t in rows]
    else:
        lines = ["| Document | Covers |", "|----------|--------|"]
        lines += [f"| [{n}]({n}) | {cell(s)} |" for n, s in rows]
    return "\n".join(lines)


def find_blocks(text: str, readme: Path) -> list[str]:
    """The kinds whose marker pairs the README carries, in document order.

    Raises ValueError naming the README on an unknown kind, a pair that does not
    close, an end before its start, or a kind declared twice.
    """
    kinds: list[str] = []
    open_kind: str | None = None
    for match in MARKER.finditer(text):
        kind, edge = match.groups()
        if kind not in KINDS:
            raise ValueError(f"{readme}: unknown inventory kind {kind!r}")
        if edge == "start":
            if open_kind is not None:
                raise ValueError(f"{readme}: inventory:{open_kind} never closes")
            if kind in kinds:
                raise ValueError(f"{readme}: inventory:{kind} appears twice")
            open_kind = kind
        else:
            if open_kind != kind:
                raise ValueError(f"{readme}: inventory:{kind}:end without its start")
            kinds.append(kind)
            open_kind = None
    if open_kind is not None:
        raise ValueError(f"{readme}: inventory:{open_kind} never closes")
    return kinds


def markers(kind: str) -> tuple[str, str]:
    return f"<!-- inventory:{kind}:start -->", f"<!-- inventory:{kind}:end -->"


def committed(text: str, kind: str) -> str:
    """The table currently between the kind's markers, without them."""
    start, end = markers(kind)
    body = text.split(start, 1)[1].split(end, 1)[0]
    return body.strip("\n")


def splice(text: str, kind: str, table: str) -> str:
    """The text with `table` between the kind's markers, replacing what was there."""
    start, end = markers(kind)
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n\n{table}\n\n{end}{tail}"


def discover(root: Path) -> list[Path]:
    """Every README.md under root that carries an inventory marker."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in PRUNE_DIRS)
        if "README.md" in filenames:
            readme = Path(dirpath) / "README.md"
            if MARKER.search(readme.read_text(encoding="utf-8")):
                found.append(readme)
    return found


def process(readme: Path, root: Path, check: bool, out: Printer) -> int:
    """Regenerate (or, with check, verify) every table one README carries."""
    label = readme.relative_to(root)
    text = readme.read_text(encoding="utf-8")
    try:
        kinds = find_blocks(text, label)
    except ValueError as exc:
        out.error(f"inventory: {exc}")
        return 1
    if not kinds:
        out.error(f"inventory: {label} carries no inventory markers")
        return 1

    status = 0
    for kind in kinds:
        try:
            rows = ROWS[kind](readme.parent, root)
        except ValueError as exc:
            out.error(f"inventory: {label} {kind}: {exc}")
            out.line("  The first sentence of the source is the description column.")
            return 1
        # The empty-set guard. A directory with nothing to list would render a
        # header and no rows, and --check would then hold the README to it.
        if not rows:
            out.error(f"inventory: {label} {kind}: nothing to list")
            return 1
        table = render_table(kind, rows)
        if not check:
            text = splice(text, kind, table)
            out.ok(f"{label} {kind}: {len(rows)} rows")
            continue
        current = committed(text, kind)
        if current == table:
            out.ok(f"inventory: {label} {kind}: {len(rows)} rows")
            continue
        status = 1
        out.error(f"inventory: {label} {kind} is stale")
        diff = difflib.unified_diff(
            current.splitlines(),
            table.splitlines(),
            fromfile="committed",
            tofile="generated",
            lineterm="",
        )
        for line in diff:
            out.line(f"  {line}")
        out.line("  Solution: regenerate the table")
        out.line(f"  Command: python scripts/docs/gen_inventory.py {label}")
    if not check:
        readme.write_text(text, encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "readmes",
        nargs="*",
        help="README.md files to render (default: every README carrying a marker)",
    )
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and fail when a committed table differs",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    if args.readmes:
        readmes = [
            (p if p.is_absolute() else root / p).resolve()
            for p in map(Path, args.readmes)
        ]
        for readme in readmes:
            if not readme.is_file():
                out.error(f"inventory: no such file {readme}")
                return 1
    else:
        readmes = discover(root)
        if not readmes:
            out.error(f"inventory: no README under {root} carries a marker")
            return 1

    status = 0
    for readme in readmes:
        status |= process(readme, root, args.check, out)
    return status


if __name__ == "__main__":
    sys.exit(main())
