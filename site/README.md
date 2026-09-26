# site

The user-facing documentation site for the repository's tiers, built with Docusaurus. It is
the document a researcher reads to build and run the rust harness and to operate the interp
tier; `docs/` is the knowledge base for changing the repository, and the two do not overlap by
design. Each documented tier is one top-level section with the same skeleton, and the landing
page relates them.

## What it consumes and what consumes it

The site reads four things from the tree. The per-binary reference pages under
`docs/rust-harness/reference/` are rendered from the binaries' own `--help` by
`../scripts/docs/gen_site_reference.py`, the usage line and then one table per help section,
and committed so the site builds without cargo. The per-verb pages under
`docs/interp/reference/` are rendered from the interp CLI's argparse parser by
`../scripts/docs/gen_site_interp_reference.py`, which builds the parser without torch, and
committed for the same reason. The Formats pages restate the contract files under
`../schemas/` in words. Everything else is written by hand against the binaries, the fixtures
under `../rust-harness/fixtures/`, the interp command sources, and the runbooks under
`../docs/interp/`.

Nothing in the tree consumes the site. It is not part of any Docker image.

## Entry points

From the repository root:

```bash
make site-install    # npm ci against site/package-lock.json; needs the network once
make site-gen        # regenerate the reference pages from each binary's --help
make site-build      # build into site/build; skips with [SKIP] until site-install has run
make site-serve      # build, then serve on http://localhost:3000
make site-check      # regenerate and fail if the committed reference pages were stale
```

`make quality` runs `scripts/docs/check_site_reference.py` and
`scripts/docs/check_site_interp_reference.py`, which hold the committed reference pages to
the flag tables and the argparse literals without cargo, node, or torch;
`scripts/docs/check_site_diagrams.py` and `scripts/docs/check_site_tokens.py`, which hold
the figura diagram copies and the mirrored design tokens to the record in
`static/img/diagrams/figura.json` without a figura checkout; and then `make site-build`,
which is a no-op until the dependencies are installed.

## Layout

```text
docusaurus.config.js   docs at the site root, no blog, broken links fail the build
sidebars.js            one sidebar per section, in reading order
package.json           Docusaurus and React pinned exact; package-lock.json is the pin
src/css/custom.css     the theme: tokens shared with the blog and figura, code panels, article width
src/prism/slate.js     the code panel Prism theme, dark under both page themes
static/img/logo.svg    the navbar mark and favicon, one monochrome SVG
static/img/diagrams/   architecture figures, each a byte copy of a static diagram's SVG export from the figura repo; figura.json records the commit and sha256 of each
docs/index.md          the landing page: the tiers and how they relate
docs/rust-harness/     overview, getting-started, guides, observability, reference, troubleshooting
docs/interp/           the same skeleton for the interp tier; reference/<verb>.md is generated
docs/formats/          the checkpoint layout, four artifacts, and the capture-to-probe guide
```

## Traps

- **The reference pages are generated.** Editing one by hand lasts until the next
  `make site-gen`, and the quality gate fails when a page lacks a flag its binary has. Edit the
  flag table in the source and regenerate.
- **A new binary or verb needs a sidebar line.** The generators discover binaries from
  `../rust-harness/runtime/src/bin/` and verbs from `interp.cli.COMMAND_MODULES`, so a new
  one gets a page on the next run; `sidebars.js` is explicit, and the build fails on the
  missing entry until it has one.
- **The interp generator walks the parser, not its text.** A verb's page is what
  `argparse` knows: description, usage, one row per `add_argument`. A flag whose help is
  empty renders an empty description cell, which is a prompt to write one in the source, not
  on the page. Nested subcommands (`organism`) render as sections of one page.
- **The generator parses `--help` text.** Its three rules are the two renderers' shape: a
  column-zero line ending in a colon opens a section, a two-space line is a term, a six-space
  line continues a description. A renderer change that breaks that shape breaks the parse, and
  `../tests/docs/test_gen_site_reference.py` pins it.
- **The diagrams are copies, not sources.** An SVG under `static/img/diagrams/` is figura's
  committed export, byte for byte, and `figura.json` records the figura commit, each file's
  export path and its sha256. Editing a copy in place fails the quality gate: fix the
  diagram in figura, re-export, re-copy, and update the record. Moving to a newer figura
  commit updates the record's token snapshot from figura's `tokens.css` too, and
  `figura.json` is published with the site like any other static file.
- **The panel tokens mirror figura.** `src/css/custom.css` declares figura's panel colours,
  radius, easing and sans stack under the blog's names, and `check_site_tokens.py` compares
  each against the record's snapshot through an explicit pair table. A value changes in
  figura first. `--font-mono` differs from figura's `--fg-mono` and is listed unpaired: the
  difference is an open decision, not an oversight.
- **Pages are CommonMark, not MDX.** `markdown.format` is `detect`, so `.md` files keep
  literal angle brackets and braces. An `.mdx` file would be parsed as MDX.
- **`.gitignore` ignores every `package.json` and lockfile in the tree** for the browser
  verification harness's sake; this directory's two are carved back out by name. A second
  package directory would need the same.
- **Every output block on the site was recorded on a real Gemma 4 checkpoint**, and is taken
  verbatim from an Observed block of `../docs/verification/rust-harness/rust-harness.md`.
  The synthetic fixture appears only where it is the subject, and each
  such page says so. A command with no recorded output has no output block.
