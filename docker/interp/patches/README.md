# Vendored Neuronpedia patches

Patches in this directory apply to the pinned upstream checkout: `make np-clone` runs
`git apply` on every `*.patch` here, in filename order, after checking out the pinned SHA.
The checkout reset (`checkout -f`) plus re-apply keeps the sequence idempotent.

Rules, per the PS4 design decision:

- One construction site per patch, upstreamable by default (add an env-var override with
  the upstream value as the fallback; never change behavior when the variable is unset).
- Each patch records the upstream PR link in its header comment; the patch is deleted
  when that PR lands in a pin bump.
- A pin bump must re-verify every patch still applies (`make np-clone` fails loudly if
  not) and re-check whether upstream has made it obsolete.

Current content:

- `0001-graph-dockerfile-noble-python.patch` (KB-68): removes noble's
  `EXTERNALLY-MANAGED` marker in the graph image's `base-cuda` stage, so the Dockerfile's
  `pip install poetry` works on the Ubuntu 24.04 base that
  `docker/interp/neuronpedia.override.yaml` selects through build args. The two are a
  pair: the build args alone fail at `pip install`, the patch alone changes nothing on
  22.04. Not yet filed upstream.

The autointerp OpenRouter patch (`OPENROUTER_BASE_URL` in the client construction, see
`docs/interp/investigations/ps4_autointerp_investigation.md`) was anticipated here and never
needed: verification found upstream support.
