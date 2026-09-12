# site

The user-facing documentation site for the rust harness, built with Docusaurus. It is the
document a researcher reads to build and run the harness; `docs/` is the knowledge base for
changing the repository, and the two do not overlap by design.

## What it consumes and what consumes it

The site reads three things from the tree. The per-binary reference pages under
`docs/rust-harness/reference/` are rendered from the binaries' own `--help` by
`../scripts/docs/gen_site_reference.py`, the usage line and then one table per help section,
and committed so the site builds without cargo. The Formats
pages restate the contract files under `../schemas/` in words. Everything else is written by
hand against the binaries and the fixtures under `../rust-harness/fixtures/`.

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

`make quality` runs `scripts/docs/check_site_reference.py`, which holds the committed
reference pages to the flag tables without cargo or node, and then `make site-build`, which
is a no-op until the dependencies are installed.

## Layout

```text
docusaurus.config.js   docs at the site root, no blog, broken links fail the build
sidebars.js            the two sidebars, in reading order
package.json           Docusaurus and React pinned exact; package-lock.json is the pin
src/css/custom.css     the theme: accent, heading sizes, article width, fonts, sidebar weights
static/img/logo.svg    the navbar mark and favicon, one monochrome SVG
docs/index.md          the overview
docs/rust-harness/     getting-started, guides, observability, reference, troubleshooting
docs/formats/          the checkpoint layout, four artifacts, and the capture-to-probe guide
```

## Traps

- **The reference pages are generated.** Editing one by hand lasts until the next
  `make site-gen`, and the quality gate fails when a page lacks a flag its binary has. Edit the
  flag table in the source and regenerate.
- **A new binary needs a sidebar line.** The generator discovers binaries from
  `../rust-harness/runtime/src/bin/`, so a new one gets a page on the next run; `sidebars.js`
  is explicit, and the build fails on the missing entry until it has one.
- **The generator parses `--help` text.** Its three rules are the two renderers' shape: a
  column-zero line ending in a colon opens a section, a two-space line is a term, a six-space
  line continues a description. A renderer change that breaks that shape breaks the parse, and
  `../tests/docs/test_gen_site_reference.py` pins it.
- **Pages are CommonMark, not MDX.** `markdown.format` is `detect`, so `.md` files keep
  literal angle brackets and braces. An `.mdx` file would be parsed as MDX.
- **`.gitignore` ignores every `package.json` and lockfile in the tree** for the browser
  verification harness's sake; this directory's two are carved back out by name. A second
  package directory would need the same.
- **Every command on the site is run in the sandbox before it is written down**, against the
  committed synthetic checkpoints. A page that needs the real checkpoint or the workstation
  says so in the page.
