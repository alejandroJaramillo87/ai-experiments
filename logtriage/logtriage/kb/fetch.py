"""Fetching the MongoDB Database Manual, and recording exactly what was fetched.

The manual now lives inside a documentation monorepo holding every MongoDB docs
product -- Atlas, every driver, Compass, the shell, the tools. Cloning that
whole tree to read one product would be wasteful anywhere and is a session-ending
mistake in a sandbox with a fixed disk allowance, so this fetches the manual
alone: a blobless partial clone, then a sparse checkout of one directory.

Nothing fetched is ever committed. The manual is published under CC BY-NC-SA 3.0
US -- stated on the pages themselves rather than in a LICENSE file, the
repository has none -- and vendoring it would pull share-alike terms over this
tree. What is committed instead is the manifest: the resolved commit, and a
sha256 for every file that went into the index. That is enough to say precisely
which corpus produced a given index, which is the property that actually matters
and the only one a copy of the text would add.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from logtriage.kb.corpus import file_digest

DEFAULT_REMOTE = "https://github.com/mongodb/docs.git"
DEFAULT_BRANCH = "main"

# Where the Database Manual sits inside the monorepo. Beside it are `upcoming`
# and the frozen `v4.4`..`v8.0` trees; `manual` is the current one, and its
# snooty.toml is what reports which server version that is.
MANUAL_PATH = "content/manual/manual"

# Files that make up the corpus. The manual uses .txt for pages and .rst for the
# fragments they include; both are reStructuredText and both are needed, since
# four pages in five include something.
CORPUS_SUFFIXES = (".txt", ".rst")

# Ceiling on the checked-out tree. The manual measures about 90MB; a clone that
# lands far above that has fetched something other than what was asked for --
# in practice a server that ignored the partial-clone filter and handed back
# every docs product. Failing here costs a re-run, whereas continuing fills the
# disk and takes the session with it.
MAX_CHECKOUT_BYTES = 600 * 1024 * 1024


class FetchError(RuntimeError):
    """A fetch that cannot proceed, with the remedy in the message."""


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run git, raising a FetchError carrying stderr rather than a traceback."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        command = " ".join(["git", *args])
        raise FetchError(
            f"{command} failed ({result.returncode})\n"
            f"  {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ''}"
        )
    return result.stdout.strip()


def tree_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def clone(destination: Path, remote: str, branch: str) -> str:
    """Blobless shallow clone plus a sparse checkout of the manual.

    `--filter=blob:none` is what keeps this cheap: the clone takes the commit
    graph and no file contents, and the checkout then fetches only the blobs the
    sparse pattern actually needs. If the server does not honour the filter git
    says nothing and clones everything, so the size check afterwards is not
    belt-and-braces -- it is the only signal that the fetch went wrong.
    """
    _git(
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        "--depth",
        "1",
        "--single-branch",
        "--branch",
        branch,
        remote,
        str(destination),
    )
    # Cone mode takes directories, and includes the files sitting in their
    # ancestors -- which is how snooty.toml, a sibling of source/, arrives
    # without being named.
    _git("sparse-checkout", "set", "--cone", f"{MANUAL_PATH}/source", cwd=destination)
    _git("checkout", branch, cwd=destination)

    size = tree_bytes(destination)
    if size > MAX_CHECKOUT_BYTES:
        raise FetchError(
            f"Checkout is {size / 1e6:.0f}MB, above the {MAX_CHECKOUT_BYTES / 1e6:.0f}MB "
            f"ceiling. The partial-clone filter was probably ignored, so the whole "
            f"documentation monorepo was fetched instead of the manual.\n"
            f"  Solution: remove the checkout and retry; if it recurs, the remote "
            f"does not support blob filtering and needs a local mirror\n"
            f"  Command: rm -rf {destination} && "
            f"LOGTRIAGE_KB_SOURCE=/path/to/mirror python -m logtriage fetch-kb"
        )
    return _git("rev-parse", "HEAD", cwd=destination)


def head_commit(checkout: Path) -> str:
    """The commit a checkout is sitting on, for a manifest written over one."""
    return _git("rev-parse", "HEAD", cwd=checkout)


def corpus_files(source: Path) -> list[Path]:
    """Every file that contributes text to the index, in a stable order."""
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix in CORPUS_SUFFIXES
    )


def write_manifest(
    manifest_path: Path,
    *,
    root: Path,
    commit: str,
    remote: str,
    branch: str,
    version: str,
) -> int:
    """Record the corpus: its provenance, and a digest per file.

    The digests are the point. An index built from this corpus carries the
    commit, and the per-file hashes let a later run prove the tree it is reading
    is the tree the index was built from -- without which "rebuild the index"
    and "rebuild the index from the same inputs" are indistinguishable.
    """
    source = root / "source"
    files = corpus_files(source)
    sizes = [path.stat().st_size for path in files]
    entries: list[dict[str, Any]] = [
        {
            "path": path.relative_to(source).as_posix(),
            "sha256": file_digest(path),
            "bytes": size,
        }
        for path, size in zip(files, sizes, strict=True)
    ]
    manifest = {
        "remote": remote,
        "branch": branch,
        "commit": commit,
        "manual_path": MANUAL_PATH,
        "manual_version": version,
        "license": (
            "CC BY-NC-SA 3.0 US, asserted on the published pages; the repository "
            "carries no LICENSE file. The corpus is never committed."
        ),
        "file_count": len(entries),
        "total_bytes": sum(sizes),
        "files": entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return len(entries)


def default_kb_dir() -> Path:
    """Where the corpus lives, overridable for the machines that are not this one.

    The desktop and the MacBook get the corpus by pointing at a directory rather
    than re-cloning, which is also how an air-gapped run works.
    """
    override = os.environ.get("LOGTRIAGE_KB_SOURCE")
    return Path(override) if override else Path(".kb")
