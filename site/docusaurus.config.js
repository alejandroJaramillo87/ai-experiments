// Docusaurus configuration for the user documentation site.
//
// The site documents the repository's tiers for someone who wants to run
// them, not change them: the rust harness and the interp tier today, with a
// landing page that relates them. docs/ in the parent repository is the
// internal knowledge base and is deliberately not mounted here. Docs are served at the site root: there is
// no landing page, no blog, and no external fonts or analytics, so the build
// needs no network once node_modules is installed. The theme is
// src/css/custom.css over theme-classic, with src/prism/slate.js for code
// panels; the logo is a hand-written SVG under static/img.

// @ts-check

import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import slate from './src/prism/slate.js';

// The footer names the harness version, read from the workspace manifest so it
// cannot lag a release; the other tiers have no release version to name. A manifest without the field is a build failure rather
// than a footer reading "undefined".
const manifest = readFileSync(
  fileURLToPath(new URL('../rust-harness/Cargo.toml', import.meta.url)),
  'utf8',
);
const versionMatch = manifest.match(/^\[workspace\.package\][^[]*?^version\s*=\s*"([^"]+)"/ms);
if (!versionMatch) {
  throw new Error('rust-harness/Cargo.toml has no [workspace.package] version');
}
const harnessVersion = versionMatch[1];

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI experiments',
  tagline: 'Local inference, interpretability, and training on one workstation',
  url: 'https://example.invalid',
  baseUrl: '/',
  favicon: 'img/logo.svg',

  // A broken link is a build failure. The reference pages are regenerated from
  // the binaries, and a guide that links to a flag page that no longer exists
  // must fail here rather than 404 for a reader.
  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',

  markdown: {
    // .md files are CommonMark, not MDX. The generated reference pages carry
    // literal placeholders such as <checkpoint-dir> inside fenced blocks, and
    // hand-written pages spell them in prose; CommonMark leaves both alone,
    // where MDX would parse the angle brackets as a JSX tag and fail the build.
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'docs',
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          showLastUpdateTime: false,
        },
        blog: false,
        pages: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        // The title is the wordmark; custom.css sets it in the blog's bracketed
        // mono form, so no logo image sits beside it. The SVG stays the favicon.
        title: 'AI experiments',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'overview',
            position: 'left',
            label: 'AI experiments',
          },
          {
            type: 'docSidebar',
            sidebarId: 'harness',
            position: 'left',
            label: 'Rust Harness',
          },
          {
            type: 'docSidebar',
            sidebarId: 'formats',
            position: 'left',
            label: 'Formats',
          },
        ],
      },
      footer: {
        style: 'light',
        links: [
          {
            label: 'Source',
            href: 'https://github.com/alejandroJaramillo87/ai-experiments',
          },
        ],
        copyright: `Generated from the ai-experiments repository. Rust harness ${harnessVersion}.`,
      },
      prism: {
        // One dark theme under both page themes: code panels share the dark
        // slate material of the blog's code blocks and figura's diagrams,
        // and read as framed figures on the light page.
        theme: slate,
        darkTheme: slate,
        additionalLanguages: ['bash', 'json', 'rust', 'yaml'],
      },
    }),
};

export default config;
