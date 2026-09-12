// Docusaurus configuration for the user documentation site.
//
// The site documents the rust harness for someone who wants to run it, not
// change it. docs/ in the parent repository is the internal knowledge base and
// is deliberately not mounted here. Docs are served at the site root: there is
// no landing page, no blog, and no external fonts or analytics, so the build
// needs no network once node_modules is installed.

// @ts-check

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'rust harness',
  tagline: 'An instrumented local inference runtime and coding agent for Gemma 4',
  url: 'https://example.invalid',
  baseUrl: '/',
  favicon: undefined,

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
        title: 'rust harness',
        items: [
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
        style: 'dark',
        copyright: 'Generated from the ai-experiments repository.',
      },
      prism: {
        additionalLanguages: ['bash', 'json', 'rust', 'yaml'],
      },
    }),
};

export default config;
