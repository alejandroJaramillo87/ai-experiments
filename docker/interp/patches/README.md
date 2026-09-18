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
- `0002-inference-dockerfile-python-version.patch` (KB-76): adds a `PYTHON_VERSION`
  build arg (upstream default 3.10) to the inference image's `base-cuda` stage and
  clears the same marker. The override passes 3.12 on the 24.04 base, because the
  inference project requires Python 3.11 or newer and upstream installs 3.10. Not yet
  filed upstream.
- `0003-inference-dockerfile-uv.patch` (KB-77): replaces the inference image's poetry
  install with a pinned uv, copies `uv.lock` and `vendor/` beside the project file, and
  runs `uv sync --frozen --no-dev` into `/opt/venv` (first on `PATH`, outside the directory
  upstream's dev overlay bind mounts). Upstream moved the project to uv without touching
  its Dockerfile, so poetry could not resolve the path and git sources. Not yet filed
  upstream, and there is nothing to file against: the next upstream commit touching the
  file deletes every Dockerfile in the repository.

0002 and 0003 are one stale Dockerfile seen twice: the interpreter defect hid the packaging
one until it was fixed.

The autointerp OpenRouter patch (`OPENROUTER_BASE_URL` in the client construction, see
`docs/interp/investigations/ps4_autointerp_investigation.md`) was anticipated here and never
needed: verification found upstream support.
