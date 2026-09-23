# interp/commands

One module per verb of `python -m interp`. Each exposes `configure(sub)`, which adds its
subparser, and a handler the parser binds; `interp/cli.py` lists them in `COMMAND_MODULES`
and `common.obs.dispatch` runs whichever the arguments name.

The handlers are thin: argument parsing, the `Printer` lines, and a call into the package
that owns the work (`probes/`, `sae/`, `steering/`, `organisms/`, or `interp/harvest.py`).
`runs.py` is the exception with no package behind it; it hands the `runs` verb to
`common.obs.runs`, which every CLI that writes the catalog shares. The site's reference
page for each verb is rendered from the parser here by `make site-gen`, and
`scripts/docs/check_site_interp_reference.py` fails when the two disagree.

## Modules

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `harvest.py` | Harvest residual-stream activations to NVMe (interp Phase 4 P4-D2). |
| `interp_gc.py` | Delete validated activation captures, keeping manifests (P4-D2 retention). |
| `organism.py` | Model-organism pipeline driver (interp Phase 4 P4-D1). |
| `probe_capture.py` | Capture labeled activations for probe training (W2). |
| `probe_eval.py` | Regenerate a probe's evaluation report with black-box baselines (W2). |
| `probe_export.py` | Export a registered probe as the safetensors image the harness loads (W3). |
| `probe_monitor.py` | Aggregate a window of served-request logs into a probe report (W4). |
| `probe_train.py` | Train per-layer probes from a capture and register the best (W2). |
| `runs.py` | List recent runs from the observability catalog (roadmap OBS-9). |
| `sae_eval.py` | Evaluate a trained SAE against its capture (interp Phase 4 P4-D3). |
| `sae_train.py` | Train an SAE from a P4-D2 activation capture (interp Phase 4 P4-D3). |
| `smoke_test.py` | Interpretability workbench smoke test (interp Phase 0 I1). |
| `steer_extract.py` | Extract a steering direction and export it as control-vector safetensors (W1). |
| `steer_scorecard.py` | Run the three-arm steering scorecard against the live services (W1). |

<!-- inventory:modules:end -->

## Where the real docs live

- [../README.md](../README.md) — the verb table with each module's make target
- [../../docs/interp/writeback.md](../../docs/interp/writeback.md) — the probe and steering verbs in use
