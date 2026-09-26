# tests/interp

The unit and branch tests for the interpretability tier in `interp/`: the probe toolkit
and its capture contract, activation harvesting and SAE training, steering, organisms
with their seal and locality checks, the report envelope, and every CLI handler driven in
process through the real parser. Everything runs CPU-only without network, GPU or
services.

`conftest.py` is what makes that possible. `fake_capture()` is a deterministic stand-in
for the hooked bf16 capture: class-0 and class-1 texts get linearly separable activations
plus text-seeded noise, so probes train to high AUROC and repeated captures are
bit-identical. `labeled_dataset`, `capture_dir` and `probe_dir` build the pipeline's
artifacts on top of it in `tmp_path`, and the autouse `train_lock_path` fixture points the
global run lock at a per-test path so tests never contend on `/mnt/ai-data`. The SAELens
runner and the serving client are replaced by injected callables where a test reaches
them.

Two things a reader expects elsewhere are here on purpose. Six modules load recipes
through `training/recipes/loader.py`, because interp recipes live under
`training/recipes/interp/` and the loader is the one definition of a recipe. And three
modules test docker and script subjects rather than the package: the interp image's
dependency set, the writable-state directories of both GPU images, and the concurrency
preflights in `scripts/interp/`.

<!-- inventory:tests:start -->

| File | Tests | Covers |
|------|-------|--------|
| `test_capture_contract.py` | 6 | The Python capture writer against the cross-language contract. |
| `test_capture_guards.py` | 7 | Capture contract guards (Campaign 2, E-22/E-23). |
| `test_capture_loader.py` | 3 | What the probe capture reads off a hooked decoder block. |
| `test_capture_model_hash.py` | 13 | The capture's record of which weights it read. |
| `test_cli_handlers_organism.py` | 21 | interp organism CLI handlers (interp/commands/organism.py): generate, verify, seal, and check-seal driven through build_parser() with real tmp-dir pipeline execution. |
| `test_cli_handlers_probes.py` | 18 | interp probe CLI handlers (interp/commands/probe_{capture,train,eval, export,monitor}.py) driven through build_parser() with real tmp-dir pipeline execution. |
| `test_cli_handlers_sae.py` | 20 | interp SAE-tier CLI handlers (interp/commands/{harvest,interp_gc,sae_train, sae_eval}.py) driven through build_parser() with real tmp-dir pipeline execution. |
| `test_cli_handlers_steering.py` | 18 | CLI handler tests for steer-extract and steer-scorecard (interp Phase 3 W1). |
| `test_evaluate.py` | 4 | W2 evaluation-report tests with injected generate/judge callables: the black-box baselines are always present and the deployable verdict follows the comparison. |
| `test_families.py` | 17 | interp/families.py: the per-family hook-point registry (WI-6). |
| `test_harvest.py` | 57 | P4-D2 activation harvesting: config validation, preflight, manifest, resume, retention. |
| `test_harvest_runner_branches.py` | 12 | P4-D2/P4-D3 runner-internal branches left uncovered by the main suites: the check_hook_dims forward-hook path, the SAELens runner/trainer config mapping (sae_lens injected via sys.modules), validation problems without a dedicated test, the real-watchdog envelope around both runs, tracker wiring, and gc_captures skip/loose-file branches. |
| `test_image_dependencies.py` | 5 | The interp image's dependency set must cover what the interp CLI imports. |
| `test_image_writable_state.py` | 1 | Every writable-state dir in the GPU images must land outside the rootfs. |
| `test_interp_cli.py` | 7 | Unified interp CLI: parser wiring only (no heavy deps executed). |
| `test_interp_tail_branches.py` | 29 | Tail-branch tests for the interp modules' remaining uncovered paths: organism generation internals (template loading edges, prompt writing, teacher-reply parsing, corpus-blinding refusals), scorecard perplexity recording and default-controller orchestration, and the monitor's service polling and log-window edges. |
| `test_monitor.py` | 20 | W4 monitor tests: the report is built from the scores the serving process recorded, malformed records are skipped rather than fatal, two runs over the same window are deterministic with a correct delta, a report is never deltaed against one produced a different way, and the max-records cap and retention policy behave as documented. |
| `test_organisms.py` | 57 | P4-D1 organism tests: sealing round-trip and tamper/leak refusal, implant and locality check logic (including the null-control inversion — the harness must fail a zero-implant organism), template rendering and corpus blinding, and organism recipe validation. |
| `test_preflight_scripts.py` | 4 | Tests for the interp concurrency preflights. |
| `test_probe_export.py` | 5 | Probe export round-trip (rust in-band monitor artifact). |
| `test_probes.py` | 19 | W2 probe toolkit tests: split determinism, registry hash refusal, and score.py parity between the from-activations and from-texts paths (the W2 acceptance list), plus the regenerable-cache property on a fake capture. |
| `test_report_envelope.py` | 5 | Report-envelope conformance (WI-12): every interp report emitter's output validates against the shared header convention in common/reports.py. |
| `test_sae.py` | 44 | P4-D3 SAE training and evaluation: config validation, capture refusal gate, manifest chaining, retention handshake, registry, core-metric math. |
| `test_saelens_contract.py` | 15 | Pins the SAELens internals chunked capture depends on (KB-58). |
| `test_steering.py` | 36 | W1 steering tests: deterministic safetensors export (re-export reproduces the same hash), wiring validation and path mapping, and scorecard orchestration with a fake service controller (interrupted scorecard leaves the service unsteered). |
| `test_weight_stats.py` | 10 | The shape statistics behind RH.1a check 1, proved on synthetic draws. |

<!-- inventory:tests:end -->

The workstation halves of the capture, probe, steer and organism chain are in
[../hardware/README.md](../hardware/README.md), and the tier's living documentation in
[../../docs/interp/](../../docs/interp/).
