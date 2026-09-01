"""Fetch the MongoDB Database Manual that backs the knowledge base.

This is the one command that needs a network, and it needs it once. Everything
downstream -- indexing, retrieval, triage -- runs against the checkout it leaves
behind, so a machine that has fetched the corpus can be taken offline and stay
useful, which is the point of the whole tool.

The corpus is never committed; `logtriage/kb-manifest.json` is. See kb/fetch.py
for why.
"""

from __future__ import annotations

import argparse
import json


def configure(sub) -> None:
    p = sub.add_parser(
        "fetch-kb",
        help="fetch the MongoDB Database Manual and write the corpus manifest",
        description=__doc__,
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="re-fetch over an existing checkout instead of refusing",
    )
    p.add_argument(
        "--remote",
        default=None,
        help="git remote to clone (default: the public mongodb/docs repository)",
    )
    p.add_argument(
        "--branch",
        default=None,
        help="branch to clone (default: main, the documentation monorepo)",
    )
    p.add_argument(
        "--manifest",
        default="kb-manifest.json",
        help="where to write the corpus manifest (default: kb-manifest.json)",
    )
    p.add_argument("--json", action="store_true", help="emit the result as JSON")
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    import shutil
    from pathlib import Path

    from logtriage.cli import LogtriageError
    from logtriage.kb.corpus import Corpus
    from logtriage.kb.fetch import (
        DEFAULT_BRANCH,
        DEFAULT_REMOTE,
        MANUAL_PATH,
        FetchError,
        clone,
        default_kb_dir,
        head_commit,
        tree_bytes,
        write_manifest,
    )
    from logtriage.output import Printer

    out = Printer(quiet=args.json)
    remote = args.remote or DEFAULT_REMOTE
    branch = args.branch or DEFAULT_BRANCH

    kb_dir = default_kb_dir()
    # A LOGTRIAGE_KB_SOURCE pointing straight at a manual checkout is used as
    # it stands. That is how a machine with no network -- the MacBook, an
    # air-gapped host -- gets the corpus: copy the directory, set the variable.
    if (kb_dir / "source").is_dir():
        manual_root = kb_dir
        commit = "local"
        out.info(f"Using the local corpus at {manual_root} (LOGTRIAGE_KB_SOURCE)")
    else:
        checkout = kb_dir / "docs"
        if checkout.exists() and args.refresh:
            out.run(f"Removing the existing checkout at {checkout}")
            shutil.rmtree(checkout)

        if checkout.exists():
            # Reused rather than refused. Rewriting the manifest over a corpus
            # already on disk is the common case -- after a parser change, or to
            # confirm the tree still hashes to what the manifest claims -- and
            # making that require a re-clone would make the cheap check the
            # expensive one.
            commit = head_commit(checkout)
            out.info(f"Reusing the checkout at {checkout} (--refresh to re-clone)")
        else:
            out.run(f"Cloning {MANUAL_PATH} from {remote} ({branch})")
            checkout.parent.mkdir(parents=True, exist_ok=True)
            try:
                commit = clone(checkout, remote, branch)
            except FetchError as exc:
                raise LogtriageError(str(exc)) from exc
        manual_root = checkout / MANUAL_PATH

    if not (manual_root / "source").is_dir():
        raise LogtriageError(
            f"No source/ directory under {manual_root}\n"
            "  Solution: the monorepo layout has moved; check where the manual "
            "now sits and pass --branch, or set LOGTRIAGE_KB_SOURCE to a "
            "manual checkout\n"
            f"  Command: git -C {manual_root.parent} ls-tree --name-only HEAD"
        )

    corpus = Corpus(manual_root)
    manifest_path = Path(args.manifest)
    count = write_manifest(
        manifest_path,
        root=manual_root,
        commit=commit,
        remote=remote,
        branch=branch,
        version=corpus.version,
    )
    size = tree_bytes(manual_root)

    if args.json:
        print(
            json.dumps(
                {
                    "root": str(manual_root),
                    "commit": commit,
                    "remote": remote,
                    "branch": branch,
                    "manual_version": corpus.version,
                    "files": count,
                    "bytes_on_disk": size,
                    "manifest": str(manifest_path),
                },
                indent=2,
            )
        )
        return 0

    out.ok(f"Manual {corpus.version} at {commit[:12]} ({size / 1e6:.0f}MB on disk)")
    out.info(f"Corpus files: {count}", indent=2)
    out.info(f"Manifest: {manifest_path}", indent=2)
    out.info("Next: python -m logtriage index-kb", indent=2)
    return 0
