// Prism theme for the code panels: figura's slate panel as the surface, with
// the token colors of Chroma's github-dark style, which is what the blog uses
// (curiosity-chronicles, assets/scss/_chroma.scss). One theme serves both page
// themes because the panel stays dark on the light page by design.

/** @type {import('prism-react-renderer').PrismTheme} */
const slate = {
  plain: {
    color: '#e6edf3',
    backgroundColor: '#0f172a',
  },
  styles: [
    {
      types: ['comment', 'prolog', 'doctype', 'cdata'],
      style: {color: '#8b949e', fontStyle: 'italic'},
    },
    {
      types: ['keyword', 'operator', 'namespace', 'important'],
      style: {color: '#ff7b72'},
    },
    {
      types: ['string', 'char', 'number', 'regex', 'inserted-sign'],
      style: {color: '#a5d6ff'},
    },
    {
      types: ['function', 'decorator', 'attr-name', 'selector'],
      style: {color: '#d2a8ff'},
    },
    {
      types: ['class-name', 'maybe-class-name', 'symbol'],
      style: {color: '#f0883e'},
    },
    {
      types: ['tag', 'entity'],
      style: {color: '#7ee787'},
    },
    {
      types: ['variable', 'constant', 'builtin', 'boolean', 'property', 'atrule', 'url'],
      style: {color: '#79c0ff'},
    },
    {
      types: ['punctuation'],
      style: {color: '#e6edf3'},
    },
    {
      types: ['deleted'],
      style: {color: '#ffa198'},
    },
    {
      types: ['inserted'],
      style: {color: '#56d364'},
    },
  ],
};

export default slate;
