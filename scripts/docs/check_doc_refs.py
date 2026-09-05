#!/usr/bin/env python3
"""Fail the quality gate when a document cites a repository path that does not exist.

Documents assert facts; nothing checked the assertions; the tree moved. The
class of defect this catches is the dull, mechanical majority: a file that was
renamed or moved while four documents kept naming the old path. One such
reference sat in AGENTS.md -- loaded into every agent session -- pointing at a
C++ source that had moved to hugepages/ long before, and it had already been
found once by a documentation audit and written into a report where nothing
acted on it. A finding in a report rots; a finding in a gate fails the build.

What this deliberately does not check: whether a stated port is *wrong*, whether
prose contradicts a design, whether a number is stale. Those need a reader. This
converts only the "names something that is not there" category into an exit code.

Scope and exemptions are decisions, recorded in EXEMPT_PARTS and EXEMPT_SUFFIXES
below rather than discovered by whoever next reads the output.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

# Directories that never hold documents describing this tree.
PRUNE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "__pycache__",
    ".ruff_cache",
    # pytest's tmp_path base when the default temp dir is noexec (the agent
    # sandbox; see tests/conftest.py). Holds generated fixtures, not documents.
    ".pytest-tmp",
}

# Exempt by path component. Investigation and verification write-ups are dated
# records: their value is that they capture the tree as it was on the day they
# were written, so a reference that has since moved is accurate history rather
# than drift. vendor/ is third-party and not ours to correct. archive/ holds
# closed campaign logs moved out of a live runbook -- same argument as
# investigations/, and the reason a directory rather than a suffix: a log is
# named after the campaign it records, not after being frozen.
EXEMPT_PARTS = ("investigations", "vendor", "archive")
EXEMPT_SUFFIXES = ("_investigation.md",)

# Empty on purpose, and kept as a mechanism rather than deleted. docs/verification/
# was exempt here until 2026-08-08, justified by the deliberately stale pre-fix
# paths in the ledger. Measured, that justification did not hold: the frozen
# ledger-archive.md contributes no dead paths at all, because its staleness lives
# in :NNN line suffixes that LINE_SUFFIX strips before resolution, and the one
# genuinely stale document (the Gemma 4 conformance record) is already exempt
# twice over through EXEMPT_PARTS and EXEMPT_SUFFIXES. The exemption bought
# nothing and cost coverage of the runbook, which is the most path-dense
# documentation in the tree and the documentation a human trusts most. Removing
# it surfaced four references, all recorded in KNOWN_ABSENT below.
#
# Non-empty again as of 2026-08-14, for agent memory rather than for the
# runbook. A triage agent's project note records what it concluded on a given
# day, including the paths it read; it is an agent's own record and not
# documentation of this tree, so a later move makes it inaccurate history
# rather than drift to repair. check_retired_names.py has carried exactly this
# prefix since it was written, and the two gates disagreeing about whether
# agent memory is in scope was itself the defect -- surfaced when the
# verification directory grew tier subdirectories and two notes citing the old
# flat paths failed one gate but not the other.
EXEMPT_PREFIXES: tuple[str, ...] = (".claude/agent-memory/",)

# Exempt by name, each for a stated reason.
EXEMPT_FILES = {
    # The published Gemma 4 architecture reference, citing the same upstream
    # transformers and llama.cpp sources for the same reason.
    "docs/models/gemma-4.md",
    # Describes the layout of the /mnt/ai-data volume, whose directories are
    # named relative to that mount rather than to the repository.
    "docs/os/storage-configuration.md",
}

# Absolute paths that belong to the host or the container, not the repository.
# A leading / is otherwise read as repo-root-relative, per
# .styles/DOCUMENTATION_STYLE.md ("File paths are absolute from repo root").
HOST_PREFIXES = (
    "/mnt/",
    "/work",
    "/tmp",
    "/opt/",
    "/home/",
    "/etc/",
    "/proc/",
    "/sys/",
    "/usr/",
    "/dev/",
    "/var/",
    "/app/",
    "~",
)

# A shell or glob metacharacter means the token is a pattern or a command
# fragment, not a claim that one specific file exists.
METACHARS = set("*<>${}|()[]!?")

# References that name something absent on purpose. Each is a (document, token)
# pair with a reason, because "this one is fine" is a decision a later reader
# must be able to audit rather than infer. Three kinds qualify: a document
# asserting that a path does *not* exist, a deliberate example of something that
# must never be added, and an artifact created at run time rather than committed.
# A path that is merely planned does not qualify unless the document is a roadmap
# declaring it as scope. A fourth kind was added when the Tier 3 stubs were
# deleted: a dated campaign ledger naming a file that has since been removed,
# where correcting the name would falsify the record of what was scoped on the
# day. That is the same argument check_retired_names.py records for exempting
# the same document, and it is entered here per reference rather than per
# document so the ledger's remaining paths stay checked.
KNOWN_ABSENT = {
    # Asserts the absence: "The workspace carries no ... .cargo/config.toml".
    (".styles/RUST_STYLE.md", ".cargo/config.toml"),
    # A crate-relative convention ("binaries live in src/bin/"), not a claim
    # about one location; every crate in the workspace has its own.
    (".styles/RUST_STYLE.md", "src/bin/"),
    # Roadmap scope line: the deliverable *is* writing this document. OBS-14's
    # entry was deleted when docs/observability/README.md was written; OBS-13
    # keeps its exemption until debugging-runs.md exists.
    (
        "docs/observability/design-and-roadmap.md",
        "docs/observability/debugging-runs.md",
    ),
    # A hypothetical malicious plugin, used to describe what the sandbox blocks.
    ("docs/sandbox/agent-sandbox.md", ".opencode/plugin/evil.ts"),
    ("docs/agents/tooling-parity.md", ".opencode/plugin/evil.ts"),
    # Written at run time by the usage ledger; .gitignore excludes .agent-state/.
    ("docs/agents/README.md", ".agent-state/usage.jsonl"),
    # Per-checkout Claude Code settings; .gitignore excludes it, and the passage
    # citing it says so outright ("Only the first is version-controlled"). Its
    # absence from a fresh clone is the property being documented.
    ("docs/agents/README.md", ".claude/settings.local.json"),
    ("docs/agents/tooling-parity.md", ".agent-state/usage.jsonl"),
    # Evaluation output directory, created by the run that writes into it.
    (".claude/skills/training-runs/SKILL.md", "logs/eval/"),
    # Generated on the workstation from the reference tokenizer; the sandbox
    # cannot reach it, which is why the test it feeds is #[ignore]d.
    ("docs/rust-harness/runtime-design.md", "fixtures/template_goldens.json"),
    # Scoped by a dated QA campaign against the file as it stood; the Tier 3
    # stubs were deleted in P9.7 and replaced by runtime/src/ladder.rs.
    (
        "docs/verification/rust-harness/rust-harness-qa.md",
        "rust-harness/runtime/tests/eval_acceptance.rs",
    ),
    # The postmortem command creates this log if it is absent.
    (".claude/commands/postmortem.md", "docs/research-log.md"),
    (".opencode/commands/postmortem.md", "docs/research-log.md"),
    # Written by `make agent-export` (SESSIONS_DIR); nothing commits it.
    (".claude/commands/postmortem.md", "logs/sessions/"),
    (".opencode/commands/postmortem.md", "logs/sessions/"),
    ("docs/agents/tooling-parity.md", "logs/sessions/"),
    # A statement about where OpenCode *also* looks, explaining why authoring in
    # .claude/skills/ alone serves both tools. Its absence is the point.
    ("docs/agents/tooling-parity.md", ".opencode/skills/"),
    # An illustrative link inside a fenced markdown example demonstrating link
    # syntax. The target is invented for the example and names nothing.
    (".styles/DOCUMENTATION_STYLE.md", "../optimizations/gpu/gpu-optimizations.md"),
    # The four references surfaced by ending the docs/verification/ exemption.
    # Installed-package paths inside the KB-59 row, naming the two site-packages
    # files whose interaction produced the defect. They describe the container's
    # environment, not this tree.
    ("docs/verification/bugs/known-bugs.md", "torch/_native/registry.py"),
    ("docs/verification/bugs/known-bugs.md", "triton/runtime/build.py"),
    # The same unwritten OBS-13 deliverable already exempted for the roadmap
    # above. The runbook step names it because the step is what would confirm it.
    (
        "docs/verification/observability/observability.md",
        "docs/observability/debugging-runs.md",
    ),
    # Created on the host by the flight recorder at run time; nothing commits it.
    ("docs/verification/rust-harness/rust-harness.md", "sessions/rust-harness/"),
}

BACKTICKED = re.compile(r"`([^`\n]+)`")
# Markdown link targets. A relative link is the other half of the drift problem
# and a stricter case than a backtick: [text](path) states outright that the
# path is reachable from this document, so it is resolved against the document's
# directory only, and needs none of the is-this-even-a-path heuristics.
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")
HAS_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,6}$")

# Bases a relative reference may be written against. Documents inside a
# subproject routinely cite paths relative to that subproject's root.
EXTRA_BASES = ("rust-harness", "docs", "scripts", "hugepages")


def is_exempt(rel_path: str) -> bool:
    """True when a document is out of scope, per the exemptions documented above."""
    parts = Path(rel_path).parts
    return (
        any(p in EXEMPT_PARTS for p in parts)
        or rel_path.startswith(EXEMPT_PREFIXES)
        or rel_path.endswith(EXEMPT_SUFFIXES)
        or rel_path in EXEMPT_FILES
    )


def iter_docs(root: Path) -> list[Path]:
    """Every in-scope markdown file, enumerated without git.

    git is unusable in a contained agent sandbox -- /work is a linked worktree
    whose gitdir sits outside the bind mount -- and a `git ls-files` on the left
    of a pipe once made the shellcheck step lint zero files while reporting
    success. Walk the tree instead.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for name in filenames:
            if not name.endswith(".md"):
                continue
            path = Path(dirpath) / name
            if not is_exempt(str(path.relative_to(root))):
                found.append(path)
    return sorted(found)


def looks_like_path(token: str) -> bool:
    """True when a backticked token claims that a specific repository file exists.

    Every rejection below answers a false positive measured against the real
    tree, so none is speculative:

    - `:` survives the line-suffix strip only in a container image tag
      (`nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`).
    - `=` marks an assignment (`LD_PRELOAD=/app/lib.so`), not a path.
    - A leading `..` is relative to something the document names in prose,
      which this script cannot resolve.
    - A single-segment directory (`mlruns/`, `step_X.tmp/`, `model/`) is
      usually created at run time and is not expected in the tree. Requiring
      two segments keeps `docs/rust-harness/` while dropping those.
    """
    if not token or " " in token or "://" in token or "/" not in token:
        return False
    if any(c in METACHARS for c in token):
        return False
    if token.startswith(HOST_PREFIXES) or token.startswith(".."):
        return False
    if "@" in token or "=" in token or ":" in token or token.startswith("-"):
        return False
    if HAS_EXTENSION.search(token):
        return True
    return token.endswith("/") and token.strip("/").count("/") >= 1


def extract_links(text: str) -> list[tuple[int, str]]:
    """Every (line number, relative link target) pair in one document.

    External links and in-page anchors are skipped; a bare `#fragment` names a
    heading in this document, and validating those is a different check.
    """
    links = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for target in MD_LINK.findall(line):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            links.append((lineno, target.split("#", 1)[0]))
    return [(n, t) for n, t in links if t]


def extract_refs(text: str) -> list[tuple[int, str]]:
    """Every (line number, path-like token) pair in one document."""
    refs = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for raw in BACKTICKED.findall(line):
            token = LINE_SUFFIX.sub("", raw.strip().rstrip(",.;:"))
            if looks_like_path(token):
                refs.append((lineno, token))
    return refs


def resolves(token: str, doc: Path, root: Path) -> bool:
    """True when a reference names something that exists.

    Tried in order: repo root (the documented convention, with or without a
    leading slash), the citing document's own directory, then each subproject
    root, because a document inside rust-harness/ cites runtime/src/... .
    """
    rel = token.removeprefix("./").removeprefix("/")
    if (root / rel).exists() or (doc.parent / rel).exists():
        return True
    return any((root / base / rel).exists() for base in EXTRA_BASES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    docs = iter_docs(root)
    # The shellcheck step's empty-set guard, for the same reason: a gate that
    # silently checked nothing is indistinguishable from one that passed.
    if not docs:
        out.error(f"doc-refs: enumerated zero documents under {root}")
        out.line("  The gate would pass without checking anything.")
        return 1

    dead: list[tuple[Path, int, str]] = []
    checked = 0
    for doc in docs:
        try:
            text = doc.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            out.error(f"doc-refs: {doc.relative_to(root)} is unreadable: {exc}")
            return 1
        rel_doc = str(doc.relative_to(root))
        for lineno, token in extract_refs(text):
            if (rel_doc, token) in KNOWN_ABSENT:
                continue
            checked += 1
            if not resolves(token, doc, root):
                dead.append((doc, lineno, token))
        for lineno, target in extract_links(text):
            if (rel_doc, target) in KNOWN_ABSENT:
                continue
            checked += 1
            if not (doc.parent / target).exists():
                dead.append((doc, lineno, target))

    out.info(f"doc-refs: {len(docs)} documents, {checked} path references")
    if not dead:
        out.ok("doc-refs: every referenced path exists")
        return 0

    out.error(f"doc-refs: {len(dead)} references name paths that do not exist")
    for doc, lineno, token in dead:
        out.line(f"  {doc.relative_to(root)}:{lineno}: {token}")
    out.line("  Solution: correct the path, or delete the reference if it is obsolete.")
    out.line("  A dated record may instead belong under an exemption; see this script.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
