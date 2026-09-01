# The documentation corpus

The knowledge base is the real MongoDB Database Manual, fetched from `github.com/mongodb/docs`
and never committed.

## Table of Contents

- [Why the real manual](#why-the-real-manual)
- [Fetching it](#fetching-it)
- [The licence position, and the manifest](#the-licence-position-and-the-manifest)
- [Parsing Snooty reStructuredText](#parsing-snooty-restructuredtext)
- [What the real corpus exposed](#what-the-real-corpus-exposed)
- [Chunking](#chunking)

## Why the real manual

Every reStructuredText rule in `kb/parse.py` and `kb/chunk.py` was originally proven against
hand-written fixtures written by the same person as the parser. That is a closed loop: the
fixtures encode the same assumptions as the code, so they cannot disagree with it. They agreed
about five things that were wrong, and the real manual found all five within an afternoon.

Dummy data would also have made every retrieval number meaningless, because a corpus built to be
easy to retrieve from is.

## Fetching it

```bash
$ python -m logtriage fetch-kb      # sparse clone, writes kb-manifest.json
$ python -m logtriage index-kb      # chunk, embed, write .kb/index
$ pytest -m corpus                  # retrieval round trips over the real manual
```

The manual now sits inside a documentation monorepo carrying every MongoDB docs product, at
`content/manual/manual`. `fetch-kb` takes a blobless partial clone (`--filter=blob:none`) and a
cone-mode sparse checkout of that one directory: 91MB and about five seconds, against a full
clone of everything MongoDB publishes.

Two details that are not obvious:

- **`master` is no longer a git ref.** `raw.githubusercontent.com` still serves the old
  standalone layout, and it is already stale. Anything built on it would rot silently, so the
  fetch is a clone of `main`.
- **The fetch fails on a size ceiling rather than filling the disk.** If the remote ignores the
  partial-clone filter, git says nothing and fetches the lot — on a fixed disk allowance that is
  a session-ending failure, so the size is checked against `MAX_CHECKOUT_BYTES` and the run fails
  with the remedy.

`LOGTRIAGE_KB_SOURCE` points the corpus at a local path or a mirror, which is how a second
machine is provisioned without re-cloning, and how an air-gapped run works.

## The licence position, and the manifest

Nothing fetched is committed. The manual is CC BY-NC-SA 3.0 US, **asserted on the published pages
rather than in a LICENSE file** — the repository has none — and vendoring it would pull
share-alike terms over this tree.

`kb-manifest.json` is committed instead: the remote, the branch, the resolved commit, the manual
version, and a sha256 for every one of the 4,260 corpus files. That is what makes an index
reproducible, which is the property that matters; a copy of the text would add nothing else.

## Parsing Snooty reStructuredText

The manual is not plain rST. It is Snooty-flavoured, and four of its constructs have to be
handled before the text means anything:

| Construct | Handled by | Why it matters |
|-----------|------------|----------------|
| `{+constant+}` substitutions | `kb/constants.py` | Defined in the manual's `snooty.toml`; some expand *into* roles, which fixes the order — substitute before stripping roles |
| `:role:` targets | `kb/parse.py` | Sphinx's `~` prefix means "show only the last component", so `:binary:`~bin.mongod`` must render as `mongod` — the single most likely search term in a log-triage corpus |
| `.. include::` | `kb/corpus.py` | Four pages in five pull content in this way |
| Directive options (`:copyable:`) | `kb/parse.py` | Presented as the first line of a code block, an option reads as part of the command |

Constant expansion happens in `Corpus.read` rather than in the parser, so it applies uniformly to
pages and to included fragments, and so it happens before any role is stripped. An unknown
constant is left verbatim and counted, never blanked: blanking is unrecoverable, since nobody
reading the chunk later could tell a word had been removed.

## What the real corpus exposed

Each of these was found by running the existing parser over real pages, and each now has a
regression built from the verbatim text that exposed it, in `tests/test_kb_real_constructs.py`:

- 22 of 40 code blocks on `reference/log-messages.txt` began with a `:copyable: false` option
  line presented as the first line of the command.
- 44 role targets rendered as `~bin.mongod` rather than `mongod`.
- `{+version+}`-style constants reached chunk text unexpanded.
- 78 include targets written without an extension resolved as missing, silently dropping their
  content from the pages that included them.
- Page-level field lists such as `:orphan:` became chunks whose entire content was the word
  "orphan".

Include resolution was the decisive discovery and was not anticipated. Without it, indexing
produces 15,000 plausible-looking chunks with most explanations replaced by
`(includes /includes/fact-log-redaction.rst)` — an index that looks entirely normal.

## Chunking

Chunks are bounded by `MIN_TOKENS` and `MAX_TOKENS` and carry a heading path, the source
document, anchors, and cross-references. Two rules override the size bounds:

- **A code block is never split.** A block longer than `MAX_TOKENS` produces an oversized chunk
  rather than two halves, because half a command still looks runnable.
- **A chunk always carries its heading path.** `Chunk.render` prefixes the breadcrumb, so a
  retrieved passage is interpretable without the page around it.

The measured result is in [statistics.md](statistics.md).

## Related

- [statistics.md](statistics.md) — the measured numbers, and what they are read against
- [../mongodb/search.md](../mongodb/search.md) — indexing these chunks into `mongot`
- [../design/architecture.md](../design/architecture.md) — the `KnowledgeStore` port
