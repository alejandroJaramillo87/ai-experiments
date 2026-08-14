# Sandbox-executable verification

**Operational runbook. The entries here need no GPU, no `/mnt/ai-data`, and no model data,
so a contained agent session can execute them and paste its own output as the evidence.
Keep the commands runnable, and the paths accurate — both documentation gates check this
directory, so a stale reference here fails the gate.**

**Entries: 2 verified, 0 blocked, 0 pending, of 2.**

Every other runbook in this directory verifies behavior that depends on the workstation:
the GPU, the NVMe mount, a real checkpoint. These verify behavior that depends on neither,
but that no automated suite covers either — properties needing a real browser or a real
process boundary, which `pytest` in this repository does not provide.

The distinction that puts an entry here is **what the property depends on**, not where it
happens to have been run. An entry belongs in this tier when running it on the RTX 5090
would prove nothing that running it in a container does not.

Recording protocol and status vocabulary are defined in [README.md](../README.md). None of
these entries is a GPU consumer, so the one-at-a-time rule does not apply to them.

## Preconditions

**Node.js 18 or later, `npm`, and a Chromium build**, for OBS-V5 only. The harness looks
under `/opt/pw-browsers/chromium-*/chrome-linux/chrome` — the Claude Code cloud container
layout — and falls back to the `PW_CHROMIUM` environment variable. It exits 1 with
`[FAIL] No Chromium binary found` when neither resolves, so a missing browser is a loud
failure rather than a silent one.

`npm install` fetches only `playwright-core`, pinned exact in
`scripts/verify/obs_browser/package.json`, which performs no browser download.
`node_modules/` is gitignored, so a run leaves no repository diff.

## OBS-V5 Browser harness for the run dashboard

Implementation: `scripts/verify/obs_browser/check.mjs`, rendering the output of
`common/obs/dashboard.py` and `common/obs/serve.py`.

`common/obs/viewer.js` is 425 lines that no test in `tests/` reaches. Chart creation, marker
ingestion, log filtering, the hostile `<!--<script` payload, and live extension are all
browser behavior, and this harness is the only thing that exercises any of it. uPlot is
vendored under `common/obs/vendor/uplot/` and inlined into the page, so rendering needs no
network — which is why the property is hardware-independent and the entry lives here.

Command:

```bash
$ cd scripts/verify/obs_browser && npm install && node check.mjs
```

Set `PW_CHROMIUM` explicitly when no browser sits under `/opt/pw-browsers`:

```bash
$ PW_CHROMIUM=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux/chrome | head -1) \
    node check.mjs
```

Expected signal: exit 0 and `ALL CHECKS PASSED`; all snapshot and live checks print `[OK]`.

Observed:

```
$ cd scripts/verify/obs_browser && npm install && node check.mjs
added 1 package, and audited 2 packages in 2s

found 0 vulnerabilities
=== Run dashboard browser verification ===

[OK] Wrote /tmp/obs-browser-jLvXWn/run/run.html
[OK] fixtures and snapshot built under /tmp/obs-browser-jLvXWn
[RUN] snapshot page (file://)
[OK] charts: 3 uPlot instances (loss, lr, throughput)
[OK] markers: checkpoint at step 3 and watchdog at step 5 ingested
[OK] log: all 5 rows render by default
[OK] log: search filter narrows to the one matching row
[OK] log: warning filter keeps the warning and the unknown-level row
[OK] hostile <!--<script payload renders as text, page intact
[OK] no page errors (clean)

[RUN] live page (report --serve)
[OK] live: header arrives via polling
[OK] live: appended metrics extend the chart data
[OK] live: appended event appears in the log tail

ALL CHECKS PASSED
```

Run in the Claude Code cloud container against
`/opt/pw-browsers/chromium-1194/chrome-linux/chrome`, autodetected with `PW_CHROMIUM`
unset. The snapshot was built by a real `python -m training report` invocation, which
needs no torch.

Status: VERIFIED 2026-08-14

## OBS-V9 Retention prunes an aged file and spares a live one

Implementation: `common/obs/retention.py`, called from `common/obs/core.py` `configure()`.

Retention has never deleted a file outside a test. `prune` runs only inside `configure()`,
opportunistically at startup, and viewer commands opt out, so the first time it matters on
a real log root is the thirty-first day of that root's life. This entry forces it early
with a backdated decoy and confirms the two properties that decide whether an event file
survives: age, and whether a writer still holds the shared lock.

`tests/common/test_obs_retention.py` covers the policy matrix and both liveness directions,
including across a process boundary. What it cannot cover is the operator's view — that the
window is honoured against a directory a human can list, and that `runs.jsonl` and foreign
files come through untouched.

The viewer command runs first, on purpose. Running it after the prune would prove nothing:
by then no file is old enough to be a candidate, so a viewer that pruned and a viewer that
did not would produce identical listings.

Command:

```bash
$ D=$(mktemp -d) && mkdir -p "$D/training"
$ printf '{}\n' > "$D/training/R-20250101-000000-aaaaaa.jsonl"
$ printf '{}\n' > "$D/training/runs.jsonl"
$ printf '{}\n' > "$D/training/notes.txt"
$ touch -d '90 days ago' "$D/training/R-20250101-000000-aaaaaa.jsonl" \
    "$D/training/runs.jsonl" "$D/training/notes.txt"
$ ls "$D/training/"
$ OBS_LOG_DIR="$D" OBS_RETENTION_DAYS=30 uv run --no-sync python -m training runs
$ ls "$D/training/"
$ OBS_LOG_DIR="$D" OBS_RETENTION_DAYS=30 \
    uv run --no-sync python -m training validate \
    training/recipes/reference/tiny-e2e.yaml
$ ls "$D/training/"
```

Expected signal: the aged `R-*.jsonl` survives the `runs` listing, because a viewer command
sets `run_retention=False`; it is gone after `validate`, which prunes at startup.
`runs.jsonl` and `notes.txt` survive both despite being equally aged, the first because the
catalog is exempt by design and the second because only `R-*.jsonl` is a candidate. Each
invocation leaves its own fresh event file behind.

Observed:

```
$ ls "$D/training/"
R-20250101-000000-aaaaaa.jsonl
notes.txt
runs.jsonl

$ OBS_LOG_DIR="$D" OBS_RETENTION_DAYS=30 uv run --no-sync python -m training runs
FINISHED             RUN ID                    COMMAND  EXIT    DURATION
                                                                  0.000s
$ ls "$D/training/"
R-20250101-000000-aaaaaa.jsonl
R-20260814-190008-0ddc3f.jsonl
notes.txt
runs.jsonl

$ OBS_LOG_DIR="$D" OBS_RETENTION_DAYS=30 uv run --no-sync python -m training validate training/recipes/reference/tiny-e2e.yaml
[OK] Schema: training/recipes/reference/tiny-e2e.yaml loads as a typed recipe
[OK] Validation: run tiny-e2e passes all cross-field invariants
$ ls "$D/training/"
R-20260814-190008-0ddc3f.jsonl
R-20260814-190008-d43e7e.jsonl
notes.txt
runs.jsonl
```

All four properties hold. This is the first prune observed outside a unit test.

The blank row in the `runs` table is the decoy catalog: `read_runs` accepts any JSON object,
so the placeholder `{}` written above renders as a row with every field empty rather than
being skipped. A malformed line would have been dropped silently; a well-formed but empty
one is kept. That is the reader behaving as documented, and it is visible here only because
the decoy stands in for a real catalog.

Status: VERIFIED 2026-08-14
