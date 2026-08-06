# Interpretability PS2 database and Gemma Scope import: implementation investigation

**Frozen record — 2026-07-14. Drift against the current repository is expected and
correct; this document is never updated.**

Investigation of what implementing interp Phase 2 deliverable PS2 (database and Gemma
Scope import) entails in this repository. The planning document's design is taken as
intent; where it conflicts with the repository, the repository is the source of truth and
the deviations are recorded here. PS2 is tagged orchestrate and depends only on PS1, which
is merged: the `make np-*` harness, the pinned checkout
(`docker/interp/neuronpedia-pin.txt`, SHA `e5488b2a`), the compose override, the env-delta
merge, and the platform runbook all exist. PS2 writes no import tooling: it is an
operational procedure executed through the upstream admin panel, plus measurement, a
runbook section, and a small persistence check.

## Table of contents

- [What PS2 delivers](#what-ps2-delivers)
- [Design-versus-repository deviations](#design-versus-repository-deviations)
- [Import scope and procedure](#import-scope-and-procedure)
- [Persistence and backup](#persistence-and-backup)
- [Measurement deliverable](#measurement-deliverable)
- [Explanation-search embeddings](#explanation-search-embeddings)
- [Repository changes](#repository-changes)
- [Upstream facts to re-verify at implementation time](#upstream-facts-to-re-verify-at-implementation-time)
- [Acceptance checks and host-side split](#acceptance-checks-and-host-side-split)
- [Risks](#risks)
- [Open questions](#open-questions)

## What PS2 delivers

A populated local database: the Gemma Scope residual-stream sources
(`gemmascope-res-16k`) for Gemma-2-2B imported through the webapp admin panel, browsable
as feature dashboards with working activation search on `localhost:3001`, and persistent
across `make np-down && make np-up`. The other output is data the ecosystem does not
publish: measured per-layer import wall-clock and resulting Postgres size, recorded in the
runbook against the 1–2TB interp storage budget.

PS3 (steering UI needs imported sources to browse), PS4 (autointerp explains imported
features), and PS5 (graph viewer) all sit on PS2.

## Design-versus-repository deviations

The planning document matches the merged PS1 state closely; PS1 already landed every hook
PS2 needs. Verified against the repository:

| Design says | Repository reality | Resolution |
|-------------|--------------------|------------|
| Import through the admin panel at `localhost:3001/admin` | Webapp is remapped to 3001 (`docker/interp/neuronpedia.override.yaml:19`, `NEXT_PUBLIC_URL`/`NEXTAUTH_URL` deltas in `docker/interp/neuronpedia.env-deltas`) | Consistent; no change needed |
| Persistence via the PS1 bind mount | Override maps Postgres data to `/mnt/ai-data/interp/neuronpedia/postgres` (`neuronpedia.override.yaml:28`), replacing the upstream named volume | Consistent; persistence across `np-down` is a property of the merged override, PS2 only proves it |
| `OPENAI_API_KEY` left unset by default | `env.interp.example` already documents it as optional for PS2; `scripts/interp/np_env.sh` copies it into the checkout's `.env` only when present | Consistent; leave unset, no work |
| Runbook records import time and size | `docs/interpretability/platform.md` exists and the PS1 investigation reserves a slot: "The PS2 import runbook slots in here later" (`ps1_neuronpedia_harness_investigation.md:173`) | New section in `platform.md`, not a new document |
| Import scope `gemmascope-res-16k` for gemma-2-2b(-it) | `MODEL_SOURCESET=gemma-2-2b-it.gemmascope-res-16k` is already the `np-inference-up` default (`Makefile`, `platform.md:66`) | Consistent; the imported source set matches the inference preset |

No conflicts found. PS2 is the first deliverable whose planning-doc design and repository
state are already aligned; the work is execution and documentation, not reconciliation.

## Import scope and procedure

Scope per the planning decision: `gemmascope-res-16k`, all layers, for Gemma-2-2B(-it).
Wider sets (65k widths, attention and MLP sites) are pulled later by projects that need
them, per the research-pulls-infrastructure rule.

Procedure is upstream's supported path, no custom tooling:

1. `make np-up` (webapp + postgres + db-init; CPU-only, no preflight needed).
2. Open `http://localhost:3001/admin`, pick model, then source set, then layer. The
   import downloads from the public S3 dataset dumps.
3. Because imports are not resumable, the operational rule replaces code: import one
   layer at a time, verify it appears in the dashboard, record wall-clock and database
   size, then proceed to the next layer.
4. After the final layer: `make np-down && make np-up`, confirm sources survive, run
   `make np-smoke`.

Whether the GPU inference instance must be up during import is an implementation-time
verification (dashboards need activation data; the import itself is expected to be
webapp + postgres only). If `np-inference-up` is required for dashboard rendering, the
concurrency preflight applies as usual: llama-gpu, vllm-gpu, and comfyui down first.

## Persistence and backup

Nothing new to build. The database lives in the PS1 bind mount; backup is `make np-down`,
then snapshot `/mnt/ai-data/interp/neuronpedia/postgres` (already documented in
`platform.md`). Retention policy per the planning decision is "do not be precious": the
database is regenerable at the cost of import hours, so no automated backup job. A single
manual snapshot after the full import completes is worth the disk, since it converts
"hours of non-resumable re-import" into "copy a directory" — record its size too.

## Measurement deliverable

Per-source-set database size is unpublished upstream; measuring it is part of PS2's
output. Record in the runbook, per layer:

- Import wall-clock (admin panel start to dashboard-visible).
- Postgres directory size after the layer
  (`du -sh /mnt/ai-data/interp/neuronpedia/postgres`).
- Cumulative total against the 1–2TB interp storage budget.

Sizes must be measured host-side; the agent records the reported commands and output
verbatim in the runbook table.

## Explanation-search embeddings

Feature search calls OpenAI via `OPENAI_API_KEY` for explanation-embedding search.
Decision stands: leave unset; search degrades gracefully to activation search, which is
what the acceptance test exercises. Revisit only if a project pulls semantic explanation
search, preferring a local embedding endpoint if upstream grows a base-URL override.
`scripts/interp/np_env.sh` already handles the key if it is ever set.

## Repository changes

PS2 is orchestrate-only, so the diff is small:

| Path | Change |
|------|--------|
| `docs/interpretability/platform.md` | New "Gemma Scope import" section: per-layer procedure, one-layer-at-a-time rule, verification step, measurement table (layer, wall-clock, cumulative Postgres size), post-import persistence check |
| `docs/interpretability/ps2_gemma_scope_import_investigation.md` | This document |

Explicitly not built, per the planning decisions: import scripts, resume logic, automated
backups, database size alerts. If the admin flow turns out to be unusable at the pin
(see risks), that becomes a pin-bump under the PS1 procedure, not custom tooling here.

## Upstream facts to re-verify at implementation time

From the planning document (verified 2026-07) and the PS1 investigation; re-verify
against the pinned checkout SHA `e5488b2a` before starting the import:

- Admin panel route and flow: `/admin`, pick model → source set → per-layer download from
  the public S3 dataset dumps. Confirm the exact page path and that localhost mode
  (sign-in disabled) reaches it without an account.
- Imports take tens of minutes per source set and cannot be resumed if interrupted.
- `gemmascope-res-16k` layer count for gemma-2-2b: 26 residual-stream hook points
  (layers 0–25); confirm the admin panel's per-layer listing matches.
- Whether the dashboard visibility check after each layer needs the inference instance
  (activation rendering) or reads purely from Postgres.
- Whether the import writes anything outside Postgres (for example dashboard activation
  caches on disk) that would also need to be in the persistence check.

## Acceptance checks and host-side split

All execution is host-side (running platform, browser, GPU host); the sandbox contributes
the runbook edits and records reported output verbatim.

1. Feature dashboards and activation search work against the imported Gemma Scope sources
   on `localhost:3001` — first half of the roadmap's criterion 1. Human reports the
   browser result; activation search additionally exercised via the UI.
2. Sources survive `make np-down && make np-up`. Human runs the cycle, confirms a
   previously imported dashboard still renders, and runs `make np-smoke`.
3. The runbook records measured import time and database size per layer. Verified by the
   filled measurement table in `platform.md`.

## Risks

- **Non-resumable imports.** An interrupted layer import leaves unknown partial state.
  Mitigation is the operational rule (one layer at a time, verify, snapshot after
  completion); if a layer wedges, the fallback is restoring the Postgres directory from
  the last snapshot, or worst case dropping the database and re-importing.
- **Unknown database size.** Unpublished upstream; if the full 16k set materially dents
  the 1–2TB budget, stop and record before continuing. Measuring this is the deliverable,
  not a surprise.
- **Admin flow drift.** The import UI is upstream's fastest-moving surface. If the flow
  at pin `e5488b2a` is broken, bump the pin per the `platform.md` procedure (np-smoke
  green before merge) rather than patching the checkout.
- **Disk pressure during import.** S3 downloads may stage through the checkout or
  container tmp before landing in Postgres; watch free space on `/mnt/ai-data` during the
  first layer.
- **Import hours occupy the webapp.** Not a GPU concern (import is CPU/DB), so the
  inference stack may stay up throughout unless dashboard verification needs
  `np-inference-up`.

## Open questions

Resolved with the owner (2026-07-14):

1. Model variant: import the model entry that carries the `gemmascope-res-16k` source
   set; if both base and `-it` appear, prefer the one matching the inference preset
   (`gemma-2-2b-it.gemmascope-res-16k`). Tentative — confirm against the admin panel's
   actual listing at import time and record what was imported.
2. Post-import snapshot: approved. One manual copy of the Postgres directory after the
   full import; record its size.
3. GPU during verification: acceptable if required. If dashboard verification needs the
   inference instance, import sessions serialize with other GPU work per the concurrency
   matrix.
