# Vendored uPlot

Chart library inlined into the run dashboard (roadmap OBS-11). uPlot is
canvas-based, 51KB minified, MIT-licensed, and built for dense time series,
which matches a per-step training series of 10,000 or more points.

| Property | Value |
|----------|-------|
| Version | 1.6.32 |
| License | MIT |
| Source | https://registry.npmjs.org/uplot/-/uplot-1.6.32.tgz |
| Tarball sha256 | `4b8a8191e425658e3ea8c8c1314b0fc679c861ee3a1af2b20c7b16ba50b5133d` |
| Files taken | `package/dist/uPlot.iife.min.js`, `package/dist/uPlot.min.css` |

The checksum above is also recorded in `docs/observability/roadmap.md`
(file and directory layout section); the two must match.

## Update procedure

1. Download the new npm tarball:
   `curl -sSL -o uplot-<version>.tgz https://registry.npmjs.org/uplot/-/uplot-<version>.tgz`
2. Compute `sha256sum uplot-<version>.tgz` and record it in this table and
   in the roadmap before extracting anything.
3. Extract `package/dist/uPlot.iife.min.js` and `package/dist/uPlot.min.css`
   and copy them over the files here.
4. Re-render a dashboard (`python -m training report --run <run_dir>`) and
   confirm the charts still draw before committing the bump on its own.
