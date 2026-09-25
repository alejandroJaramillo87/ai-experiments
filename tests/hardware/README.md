# tests/hardware

The workstation tier: tests that need the RTX 5090, the Ryzen 9950X, host RAM,
`/mnt/ai-data`, a real checkpoint or the built rust-harness binary, and are therefore
deselected by default and run only where those exist. Most are the workstation half of a
verification runbook entry whose sandbox half lives in `tests/training/`, `tests/interp/`
or the rust workspace; the entry id is the first word of each module's docstring.

## Tiers, markers and the rule

Every test carries exactly one tier marker, at least one resource marker and exactly one
subsystem marker, and `test_markers.py` enforces all three, plus that every `hw_*` marker appears in `addopts` so a
sandbox deselects rather than fails on collection.

| Marker | Means |
|--------|-------|
| `hw_smoke` | under five minutes total, no model downloads, temporary-directory writes only; the before-every-session gate |
| `hw_full` | may allocate near-capacity VRAM, RAM and disk and load real models; per verification phase and after an environment change |
| `hw_e4b` | needs the real Gemma 4 E4B checkpoint on disk; a provisioning precondition carried beside a tier marker |
| `gpu`, `cpu`, `ram`, `ssd` | the resource exercised |
| `harness`, `training`, `interp`, `platform` | the subsystem proved; `make hardware-tests SUITE=<subsystem>` selects on it, combined with a tier |

**A test fails rather than skips when the hardware it exists to prove is absent.** The
`cuda` fixture asserts `torch.cuda.is_available()`, and `require_binary()` asserts the
release harness was built by `make harness-build`. The one exception is a real-model load,
which skips cleanly when the weights are not on disk, because gated hub access is out of
scope by decision. Modules import without torch so sandbox collection stays clean; the
heavy imports live inside the tests.

`conftest.py` resolves the checkpoints: `HARNESS_E4B_MODEL` names the E4B directory for the
`hw_e4b` tier, else `GEMMA4_CHECKPOINT_ROOT` (default `/mnt/ai-data/models/hf`) is the root
both `e4b_checkpoint()` and `e2b_it_checkpoint()` read. `default_root_env()` strips
`OBS_LOG_DIR` so the binary resolves its default observability root, which is a fact the
record suites check.

## Running

```bash
uv run --no-sync pytest -m hw_smoke tests/hardware/ -v     # session-start gate
uv run --no-sync pytest -m hw_full tests/hardware/ -v      # per verification phase
make hardware-tests HW=full SUITE=training                 # one tier of one subsystem
make quality-full HW=e4b HARNESS_E4B_MODEL=/mnt/ai-data/models/hf/gemma-4-E4B-it
```

Everything here is a GPU consumer, so the one-job-at-a-time rule in `AGENTS.md` applies
across every worktree. `--no-sync` is deliberate: a plain `uv run` resyncs the environment
and triggers a from-source flash-attn build. Its cost is that nothing reconciles `.venv`
with `uv.lock` either, so a pin that leaves the lock stays installed; `test_venv_lock.py`
is the smoke-tier check that names such a stray, and `make install` is the repair.

<!-- inventory:tests:start -->

| File | Tests | Covers |
|------|-------|--------|
| `test_alignment.py` | 5 | RH.19, the alignment artifact on the real vocabulary, as a test the E4B tier runs. |
| `test_attention_capture.py` | 3 | RH.36, attention-capture as its own binary on the real E4B, as a test the E4B tier runs. |
| `test_capture_run_record.py` | 4 | RHX.5, a capture's run record on real weights, as a test. |
| `test_chat_marker.py` | 4 | RHX.12, a chat at a terminal shows where the next turn goes, as a test. |
| `test_checkpoint_info.py` | 4 | RH.22, checkpoint-info --check against the real E4B, as a test the E4B tier runs. |
| `test_compaction.py` | 3 | RH.20, compaction past the rewind horizon, as a test the E4B tier runs. |
| `test_cpu.py` | 4 | CPU hardware suite (WI-3): Zen 5 feature flags and a CPUAdam step at the smoke tier; affinity placement and the CPUAdam step-time bound at the full tier. |
| `test_fine_tuning_loop.py` | 1 | FT.hardware-test, the fine-tuning loop's seam, as one test the hw_full tier runs. |
| `test_from_scratch_loop.py` | 1 | FS.0 to FS.8, less FS.5, the from-scratch loop as one hw_full test. |
| `test_gpu.py` | 10 | GPU/VRAM hardware suite (WI-3): sm_120 presence and bf16 compute at the smoke tier; near-capacity allocation, the step measurement on the tiny recipe, tolerance-based resume, family-registry loads, and the harvest hook smoke at the full tier. |
| `test_harness_endpoint.py` | 4 | RH.14, the endpoint answering the interp client, as a test. |
| `test_heldout_eval.py` | 1 | FT.heldout-eval, held-out loss and lm-eval on base and merged, as one test the hw_full tier runs. |
| `test_interp_chain.py` | 10 | RH.9a-9e, the in-band interp chain on real weights, as tests the hw tier runs. |
| `test_ladder.py` | 4 | RH.1b, the correctness ladder at Tier E, as a test the hardware tiers run. |
| `test_layer_diff.py` | 3 | RH.21, layer-diff against a transformers capture of the real E4B, as a test the E4B tier runs. |
| `test_markers.py` | 3 | Marker invariants for the hardware suites (WI-3): every test in tests/hardware/ carries exactly one tier marker (hw_smoke or hw_full), at least one resource marker (gpu, cpu, ram, ssd) and exactly one subsystem marker (harness, training, interp, platform), and every hardware-gated marker declared in pyproject.toml is deselected by addopts. |
| `test_obs_records.py` | 4 | RH.13, the run-level observability records on the workstation, as a test. |
| `test_permissions.py` | 5 | RHX.10, the permissions layer on a model that makes the calls, as a test. |
| `test_preflight.py` | 6 | RH.16, the preflight on the real checkpoint, as a test the hardware tiers run. |
| `test_ram.py` | 4 | Host-RAM hardware suite (WI-3): huge pages and pinned buffers at the smoke tier; replay-buffer residency and ZeRO-2 host-RAM headroom at the full tier. |
| `test_recipe_gates.py` | 3 | FT.recipe-gates, the two gates a run passes before the GPU, as the hw_smoke tier runs it. |
| `test_remedy_runs_as_printed.py` | 4 | RHX.6, a failure's `Command:` remedy runs as printed, as a test. |
| `test_run_header.py` | 7 | RHX.1, the run header against the real checkpoint, as a test. |
| `test_session_resume.py` | 4 | RHX.11, a resumed session says what it resumed, as a test. |
| `test_shared_kv_weights.py` | 4 | RH.1a check 1 on the real Gemma 4 checkpoints: were the shared-KV tail's key and value projections ever trained? The modeling code builds k_proj and v_proj only for layers that own a KV cache; the released weights carry them on every layer. |
| `test_ssd.py` | 6 | SSD/storage hardware suite (WI-3): mount layout and staging writability at the smoke tier; staging throughput, checkpoint round trip, the harvest disk preflight, and the D9 data_wait_ms bar at the full tier. |
| `test_status_costs.py` | 5 | RHX.2, the load and prefill lines on real weights, as a test. |
| `test_step_qlora.py` | 1 | FT.calibrate-qlora, the reference QLoRA recipe stepped on the GPU, as the hw_full tier runs it. |
| `test_train_resume.py` | 1 | FT.train-resume, a TRL fine-tune interrupted and resumed, as one test the hw_full tier runs. |
| `test_venv_lock.py` | 1 | Environment drift suite (WI-3): every distribution installed in the interpreter's site-packages is one uv.lock names at a locked version, at the smoke tier. |

<!-- inventory:tests:end -->

The pairings between these and their sandbox halves are tabulated in
[../../docs/testing/hardware-tiers.md](../../docs/testing/hardware-tiers.md); the runbooks
they mirror are under [../../docs/verification/](../../docs/verification/).
