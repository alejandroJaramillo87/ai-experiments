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

Currently expected content: `0001-autointerp-openrouter-base-url.patch`, honoring
`OPENROUTER_BASE_URL` in the autointerp client construction, authored against the pinned
SHA after verifying upstream still hardcodes the OpenRouter URL (see
`docs/interpretability/ps4_autointerp_investigation.md`). The directory being empty means
that verification found upstream support and no patch was needed.
