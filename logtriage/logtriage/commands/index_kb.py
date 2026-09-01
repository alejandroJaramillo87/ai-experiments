"""Chunk the fetched manual, embed it, and write a retrieval index.

Runs offline against whatever `logtriage fetch-kb` left on disk. The statistics
it prints are the point as much as the index is: they are the only place the
quality of 15,000 chunks is visible, and the only thing that will notice a
chunking change for the worse.
"""

from __future__ import annotations

import argparse
import json

from logtriage.embed.factory import BACKENDS


def configure(sub) -> None:
    p = sub.add_parser(
        "index-kb",
        help="chunk and embed the fetched manual into a retrieval index",
        description=__doc__,
    )
    p.add_argument(
        "--out",
        default=None,
        help="where to write the index (default: <kb dir>/index)",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="rebuild over an existing index instead of refusing",
    )
    p.add_argument(
        "--embedder",
        default="hashing",
        choices=list(BACKENDS),
        help="how to embed the chunks (default: hashing, which needs no server "
        "and has no semantics; llama needs LOGTRIAGE_EMBED_URL)",
    )
    p.add_argument(
        "--to",
        default="file",
        choices=["file", "mongo"],
        help="where the index lands (default: file, an on-disk index triage "
        "loads; mongo pushes the chunks to mongot and waits for them to be "
        "queryable)",
    )
    p.add_argument(
        "--uri",
        default=None,
        help="MongoDB connection string for --to mongo (default: LOGTRIAGE_MONGO_URI)",
    )
    p.add_argument(
        "--json", action="store_true", help="emit the corpus statistics as JSON"
    )
    p.set_defaults(func=cmd)


def _push_to_mongo(store, embedder, meta: dict, uri: str | None, out) -> dict[str, str]:
    """Copy a built index into MongoDB and wait for mongot to catch up.

    The chunks and vectors are already in hand from the file path's build, so
    this re-uses them rather than embedding the manual twice -- which on a
    served embedder is minutes of GPU time for an identical result.

    Waiting is the load-bearing half. Search index creation is asynchronous, and
    a query against an index still building returns nothing rather than an
    error, so a command that returned here would hand the operator a deployment
    that looks broken for the next few minutes.
    """
    from logtriage.store.factory import mongo_knowledge_store

    remote = mongo_knowledge_store(uri)
    pairs = list(store.items())

    out.run(f"Writing {len(pairs):,} chunks to MongoDB")
    remote.upsert([chunk for chunk, _ in pairs], [vector for _, vector in pairs])

    # Written before the wait, so an interrupted build still leaves the index
    # describing itself. A collection of chunks whose embedder is unknown
    # cannot be safely queried by anything.
    remote.save_meta(meta)

    out.run("Creating search indexes")
    remote.ensure_search_indexes(embedder.dimensions)

    out.run("Waiting for mongot to report both indexes READY")
    status = remote.wait_until_ready()
    for name, state in sorted(status.items()):
        out.ok(f"{name}: {state}")
    return status


def _render(stats, meta: dict) -> str:
    """The statistics as a human reads them, worst-case numbers first."""
    lines = [
        f"corpus {meta['manual_version']} at {meta['corpus_commit'][:12]}",
        f"  documents           {stats.documents}",
        f"  chunks              {stats.chunks}",
        f"  tokens              min {stats.token_min}  p50 {stats.token_p50}  "
        f"p95 {stats.token_p95}  max {stats.token_max}",
        f"  oversized (>MAX)    {stats.over_max_tokens}",
        f"  undersized (<MIN)   {stats.under_min_tokens}",
        f"  carrying code       {stats.with_code}",
        f"  code blocks intact  {stats.code_blocks_intact} of {stats.code_blocks}",
        "  heading depth       "
        + ", ".join(f"{d}:{n}" for d, n in stats.heading_depth.items()),
        "",
        f"  includes resolved   {stats.includes_resolved}",
        f"  includes missing    {stats.includes_missing}",
    ]
    for area, count in stats.missing_by_area.items():
        lines.append(f"      {count:5d}  {area}")
    lines += [
        "",
        f"  unknown constants   {len(stats.unknown_constants)}",
        f"  unexpanded {{+x+}}    {stats.unexpanded_substitutions}",
        f"  residual roles      {stats.residual_roles}",
        f"  chunked in          {stats.chunk_seconds:.1f}s",
        f"  embedded in         {stats.embed_seconds:.1f}s",
    ]
    return "\n".join(lines)


def cmd(args: argparse.Namespace) -> int:
    import shutil
    from pathlib import Path

    from logtriage.cli import LogtriageError
    from logtriage.embed.factory import descriptor, make_embedder
    from logtriage.kb.corpus import Corpus
    from logtriage.kb.fetch import MANUAL_PATH, default_kb_dir
    from logtriage.kb.index import build_index
    from logtriage.output import Printer

    out = Printer(quiet=args.json)

    kb_dir = default_kb_dir()
    manual_root = (
        kb_dir if (kb_dir / "source").is_dir() else kb_dir / "docs" / MANUAL_PATH
    )
    if not (manual_root / "source").is_dir():
        raise LogtriageError(
            f"No manual corpus at {manual_root}\n"
            "  Solution: fetch it first, or point LOGTRIAGE_KB_SOURCE at a copy\n"
            "  Command: python -m logtriage fetch-kb"
        )

    destination = Path(args.out) if args.out else kb_dir / "index"
    if destination.exists() and not args.refresh:
        raise LogtriageError(
            f"An index already exists at {destination}\n"
            "  Solution: pass --refresh to rebuild it\n"
            "  Command: python -m logtriage index-kb --refresh"
        )

    # Before anything is deleted, and before the corpus is touched. --refresh
    # destroys a working index, so every reason this run could fail must be
    # found first: an unreachable embedding server would otherwise leave the
    # machine with no index at all, having had one a second earlier.
    embedder = make_embedder(args.embedder)

    if destination.exists():
        shutil.rmtree(destination)

    corpus = Corpus(manual_root)
    out.run(f"Chunking {manual_root} (manual {corpus.version})")
    store, stats = build_index(corpus, embedder)

    commit = _corpus_commit(kb_dir)
    meta = {
        # Checked on load. An index and a corpus that disagree about any of
        # these produce confident answers from the wrong text.
        "corpus_commit": commit,
        "manual_version": corpus.version,
        **descriptor(args.embedder, embedder),
        "stats": stats.as_dict(),
    }
    # The on-disk index is written either way. It is what `triage --kb` loads,
    # it is what makes a build reproducible, and pushing to MongoDB without it
    # would leave the corpus statistics with nowhere to live.
    store.save(destination, meta)

    status: dict[str, str] = {}
    if args.to == "mongo":
        status = _push_to_mongo(store, embedder, meta, args.uri, out)
        meta = {**meta, "mongo_index_status": status}

    if args.json:
        print(json.dumps({**meta, "index": str(destination)}, indent=2, sort_keys=True))
        return 0

    out.line(_render(stats, meta))
    out.line()
    out.ok(f"Indexed {len(store)} chunks to {destination}")
    if status:
        out.ok(f"Search indexes queryable: {', '.join(sorted(status))}")
    if not embedder.semantic:
        out.warn(
            "Vectors are hashed, not semantic: a retrieval score here measures "
            "word overlap and says nothing about meaning."
        )
    return 0


def _corpus_commit(kb_dir) -> str:
    """The commit the corpus is at, or 'local' for a copied-in tree."""
    from logtriage.kb.fetch import FetchError, head_commit

    checkout = kb_dir / "docs"
    if not (checkout / ".git").exists():
        return "local"
    try:
        return head_commit(checkout)
    except FetchError:
        return "local"
