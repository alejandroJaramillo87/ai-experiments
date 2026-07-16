# Test coverage investigation: training and interp tiers

Audit of the sandbox test suite, its coverage of the training and interp packages, dead
and duplicated code, and readiness for the forward model policy (Gemma 4 primary, Qwen3
fallback). Findings are current as of 2026-07-16 at commit 8316e5e (PR #48 merge). The
repository state is the source of truth throughout; the roadmap document was consulted
for intended contracts and defers to the repo wherever the two disagree. Every finding
that recommends deleting or changing existing code or tests was independently
re-verified against the working tree before inclusion; two candidate findings failed
that verification and are excluded.

## Table of contents

- [Method and environment](#method-and-environment)
- [Suite inventory](#suite-inventory)
- [The four standing failures](#the-four-standing-failures)
- [Source bugs the review surfaced](#source-bugs-the-review-surfaced)
- [Redundant and low-value tests](#redundant-and-low-value-tests)
- [Skip architecture: silent loss of sandbox signal](#skip-architecture-silent-loss-of-sandbox-signal)
- [Dead code](#dead-code)
- [Duplicated code](#duplicated-code)
- [Coverage gaps and proposed tests](#coverage-gaps-and-proposed-tests)
- [Gemma 4 and Qwen3 readiness (July 2026)](#gemma-4-and-qwen3-readiness-july-2026)
- [Process recommendations](#process-recommendations)
- [Prioritized action list](#prioritized-action-list)
- [Appendix: modules below 75 percent coverage](#appendix-modules-below-75-percent-coverage)

## Method and environment

The suite was executed in a GPU-less sandbox (Python 3.11; the project targets 3.12) to
measure what the sandbox-first test strategy actually delivers. A fresh clone runs
almost nothing: 23 of 40 test files fail collection on missing imports. After
installing numpy, omegaconf, pyyaml, xxhash, tokenizers, scikit-learn, hydra-core,
fastapi, httpx, and gguf, the suite reaches its full sandbox extent:

| Metric | Value |
|--------|-------|
| Tests collected | 716 |
| Passed | 686 |
| Failed | 4 (all genuine findings, analyzed below) |
| Skipped | 26 (torch-gated 22, datatrove-gated 4) |
| Wall clock | ~20 s serial |
| Line coverage (training + interp + common) | 62.2% of 7,872 statements |

No pinned manifest of this sandbox dependency set exists anywhere in the repo; it had
to be rediscovered by trial. That is itself a finding (see
[Process recommendations](#process-recommendations)).

Analysis was performed by parallel review of every test file against the source it
exercises, a vulture-plus-grep dead-code sweep, an AST-assisted duplication sweep, a
coverage-JSON gap analysis, and a post-refactor wiring audit of PR #48. The Gemma 4
facts below were verified against `configuration_gemma4.py` on transformers main and
the official chat template mirror; pages that the container could not reach are marked
as reconstructed from search.

## Suite inventory

| Area | Files | Tests | Sandbox status |
|------|-------|-------|----------------|
| tests/training | 30 | ~460 | 3 failures; torch/datatrove skips |
| tests/interp | 9 + conftest | 195 | all pass (gguf and fastapi installed) |
| tests/agents | 5 + fixture tree | 34 | all pass |
| tests/utils | 2 | 10 | 2 failures (environment-coupled) |

Overall assessment: the suite is healthy and dense. The reviewers found exactly one
stale test, five redundant tests, and five low-value assertions across 716 tests — a
low defect rate. Determinism contracts (shard hashes, manifest reproducibility,
deterministic resume, byte-identical re-export) and refusal paths are consistently
pinned, matching the roadmap's stated strategy of sandbox unit tests for pure logic
and host-side acceptance for GPU work. The problems are structural: skip gates that
silently drop sandbox signal, CLIs that are parse-tested only, and a handful of
import-before-validate patterns that make designed refusals untestable without torch.

## The four standing failures

All four failures carry information; none is noise.

**1. `tests/training/test_offload.py::TestCheckpointFormatGuard::test_resume_rejects_torch_format_under_offload`.**
`load_checkpoint_deepspeed` imports torch (`training/checkpoint.py:195`) before the
`checkpoint_format` validation at lines 197-204, so in a torch-less environment the
guard raises `ModuleNotFoundError` instead of the designed `ValueError`. The guard body
uses only json and os; moving the import below the check is a two-line,
behavior-preserving fix and the existing test then passes. The module docstring at
`training/checkpoint.py:28` already promises lazy torch imports for exactly this
reason. The same import-before-validate pattern appears in `training/distill.py:320`
(`build_logit_cache` imports numpy/torch before its three preflight refusals) and
`training/eval/holdout.py:69` (torch imported before the empty-holdout refusal).

**2. `tests/training/test_sft_entry.py::TestPreflight::test_probe_reports_up_services`.**
Commit e28b67c (F7) added `llama-cpu` to `SERVICES` (`training/serving.py:81`) after
this test was written; `probe_inference_services` iterates all of `SERVICES`
(`training/sft.py:150-166`), so the probe now returns three services where the test
expects two. The test is stale, but updating it is not the whole story: the SFT and
distill preflights refuse to run when any probed service is up, with the message
"training and serving must not share the GPU" (`training/sft.py:356-361`,
`training/distill.py:341,448`) — yet the `SERVICES` comment added in the same commit
states llama-cpu "is not a GPU consumer, so the one-GPU-consumer policy is unaffected"
(`training/serving.py:77-80`). As merged, a CPU-only judge service left up now blocks
GPU training. Two coherent resolutions:

1. Treat blocking as intended (training also wants the CPU cores and RAM): update the
   test expectation to three services and reword the refusal message, which currently
   claims a GPU conflict that llama-cpu cannot cause.
2. Treat blocking as an unintended F7 side effect: add a `gpu` flag to `ServiceSpec`,
   filter the preflight probe to GPU consumers, and the test passes as written. This
   also fixes the parallel divergence in `interp/probes/monitor.py:33-36`, which
   hardcodes its own llama-gpu/vllm-gpu list.

Option 2 matches the written design intent; option 1 is the smaller change. This is an
owner decision — the review deliberately does not pick.

**3 and 4. `tests/utils/test_check_py_deps.py::test_pkg_version_dist_name_mapping` and `::test_json_mode_is_pure_json`.**
Both assert facts about the running environment (llama-cpp-python installed, torch
present) and therefore can never pass in the sandbox the suite is designed for,
contradicting `tests/utils/README.md` ("the contract under test is the output format,
not the host's health"). Fixes differ per test: the dist-name mapping is pure logic —
monkeypatch `importlib.metadata.version` and assert the mapped name, do not skip; the
JSON-purity test should relax line 85 to a schema assertion (`'pytorch_version' in
payload['gpu']`), moving any host-value assertion behind a `skipif` gate.

## Source bugs the review surfaced

Reading tests against source surfaced bugs the suite does not catch. Each was
independently reproduced or traced before inclusion. Ordered by severity.

### Broken entry points and guards

| # | Location | Defect |
|---|----------|--------|
| B1 | `docker-compose.training.yaml:94`, `docker/training/Dockerfile.export` | `make export-gguf` is broken at the container boundary: since commit 312c80c, `python -m training` imports `common.output` at module scope (`training/cli.py:21`), but the export service mounts only `./scripts` and `./training` and its image copies no `common/` — `ModuleNotFoundError` before argument parsing. Add a `./common:/app/common:ro` mount (and consider a build-time COPY); host-verify with `docker compose -f docker-compose.training.yaml run --rm export python -m training --help`. |
| B2 | `docker/llama-cpu/entrypoint.cpu.sh:8` | The entrypoint hard-requires `LLAMA_CPU_MODEL` in the container environment, but compose injects only the computed `MODEL_PATH` (`docker-compose.yaml:19`); no passthrough or env_file provides the variable, so llama-cpu exits at startup and discards the `MODEL_PATH` compose computed. Pre-dates PR #48. Fix: consume `MODEL_PATH` as llama-gpu's entrypoint does. Host-verify with `make cpu-up && docker compose logs llama-cpu`. |
| B3 | `interp/steering/scorecard.py:35` | `GPU_LOCK_PATH` hardcodes `/mnt/ai-data/training/.lock`, but the canonical D9 lock is `.train.lock` (`training/lock.py:25`). `check_gpu_lock` can never detect a real training run, so a steering scorecard can recreate llama-gpu while training holds the GPU. Tests pass only because they monkeypatch the constant. Delegate to `training.lock.lock_path()` / `ensure_training_idle()`. |
| B4 | `scripts/agents/guard.sh:26` | The PreToolUse guard fails open: malformed or truncated hook JSON exits 5 (jq parse error under `set -e`), and Claude Code blocks only on exit 2 — an unparseable payload allows the tool call, while the script deliberately fails closed when jq is missing. Validate the payload first (`jq empty || exit 2`). |
| B5 | `scripts/agents/guard.sh:57` | The models-dir write detector omits downloaders: `curl -o /mnt/ai-data/models/x.gguf ...` and `wget -P /mnt/ai-data/models ...` are allowed (verified). Extend the token regex with curl/wget/rsync/install/touch/truncate. |
| B6 | `scripts/agents/runpod-idle-check.sh:71` | A fractional worker count in the API response (`workersStandby: 1.0`) breaks bash integer arithmetic, aborts the loop, and the spend guardrail prints a false "no warm workers, no active spend" all-clear, exit 0 (reproduced with a fixture carrying 0.1/hr spend). Normalize with `// 0 | floor` in jq. |

### Policy and provenance guards that never fire

| # | Location | Defect |
|---|----------|--------|
| B7 | `training/data/sft.py:242` | `CHAT_TEMPLATE_VERSION` is written into every SFT manifest (`:219`) but read back nowhere — a v1-era dataset rendered with the wrong Gemma 3 framing loads silently with the same tokenizer hash, defeating the field's stated purpose (`training/families.py:38-44`). Refuse on mismatch in `load_sft_dataset`; the test is sandbox-runnable. |
| B8 | `training/distill.py:533` | The logit-distill path never resolves base identity: `_train_locked_distill` calls `build_callbacks` without `base_identity` and its emergency checkpoint meta omits `base_revision`/`base_config_sha256`, so every distilled checkpoint records nulls and the F6 base-drift export preflight is disabled for them (`training/export/preflight.py:95-99` treats null as "not recorded"). Mirror `training/sft.py:373,420-423`. |
| B9 | `training/eval/runner.py:65` | `thinking_mode` is hardcoded `"off"` for every served GGUF (also `scripts/eval/eval_ft.py:312-313`). Correct for Gemma 4 (opt-in `<|think|>`); inverted for Qwen3 hybrids, which think by default — an F7 eval of the pinned Qwen/Qwen3-1.7B records `thinking_mode="off"` while `<think>` traces flow into judge scoring, and the `MATCHED_FIELDS` confound guard cannot catch it because both sides carry the same wrong constant. `docs/training/f7_eval_wrapper_investigation.md:184` already concedes this must be per-family. |
| B10 | `training/calculator/components.py:206` | The calculator admits gemma4/gemma4_text configs with no MoE guard: a config with `enable_moe_block=True` (the 26B-A4B shape) is priced with dense-MLP formulas, violating the module's own fail-loudly policy (`:77-79`). Raise `UnknownArchitectureError`. |
| B11 | `training/data/teacher.py:323` | Resume never reconciles orphans: completed records whose keys left the prompt set persist in `data.jsonl` while the manifest records `examples=len(prompts)` — silent dataset/manifest disagreement, contradicting the `prompt_key` docstring. Reproduced. Fail loudly when completed keys fall outside the current prompt set. |
| B12 | `training/checkpoint.py:131` | Crash window between `os.rename(tmp, final)` and the `latest` symlink repoint leaves an orphan step dir; resuming from the older `latest` and re-reaching that step raises `OSError` (ENOTEMPTY) at the next checkpoint. Reproduced end to end. Remove an existing final dir before the publish rename; same window at `:175`. |

### Smaller correctness defects

| # | Location | Defect |
|---|----------|--------|
| B13 | `training/data/pipeline.py:300` | `prepare_corpus --text-field` is honored for hub sources but ignored for local JSONL file/dir sources (datatrove readers use the default `text` key; the shard step hardcodes it) — the exposed flag silently reads no text. The lite prepare-data path handles the same flag correctly. |
| B14 | `interp/harvest.py:362` | Resumed harvests record inflated throughput: wall clock spans only pending hooks while tokens count all hooks (reproduced at 1.5x with one of three hooks pre-completed). The manifest number is the recorded P4-D2 evidence. |
| B15 | `interp/harvest.py:185` | `check_hook_dims` promises `HarvestError` on any mismatch but `get_submodule` raises raw `AttributeError` on a missing module path — the multimodal-composite Gemma 4 layout risk this gate is the documented backstop for. `validate_harvest` also never bounds-checks `hook_layers`. |
| B16 | `interp/organisms/verify.py:66` | `states_verbatim`'s docstring says plural forms count as leaks ("'lighthouses' counts") but the `\b{secret}\b` regex does not match plurals — a taboo organism answering with the plural evades the zero-verbatim-leak criterion. Fix regex or docstring; add the plural test either way. |
| B17 | `training/eval/runner.py:40` | `quant_from_name` accepts lowercase quant suffixes (Qwen official GGUF naming) but returns them unnormalized, and `check_quant_match` compares case-sensitively — genuinely matched runs are refused with "Quantization mismatch". |
| B18 | `training/eval/scorecard.py:145` | An undecided win rate (all judgments malformed) renders `False` in the Improved cell instead of `-`; the "wins/losses/ties" row never renders ties. Reproduced. |
| B19 | `training/cli.py:130` | `memfit --solve` exits 0 even when nothing fits and solve returns None, printing two `[ERROR]` lines — violates the exit-code contract in the module docstring and `.styles/PYTHON_OUTPUT_STYLE.md`. Reproduced. |
| B20 | `scripts/agents/check-agent-versions.sh:88,53` | Offline, every registry lookup warns and the summary still prints `[OK] All pins current` (false all-clear, reproduced); conversely a missing ARG pin aborts the whole script on the first error under `set -e` instead of accumulating failures. |
| B21 | `scripts/utils/check_py_deps_install.py:297` | An installed distribution whose import raises (bitsandbytes without torch) is reported `[ERROR] not installed` with `version: null` — a misdiagnosis directing the operator to install a package that is present. |
| B22 | `tests/training/test_affinity.py:42` | Test bug: `test_resolve_masks_disjoint` uses the real `os.cpu_count()` and fails on any host with 10-31 logical CPUs (simulated across 2-64). Passes today only because the sandbox has 4 and the workstation 32. Monkeypatch `cpu_count` to 32. |

## Redundant and low-value tests

The suite carries very little fat. Confirmed items, each safe to act on:

| Test | Verdict | Basis |
|------|---------|-------|
| `tests/training/test_sft.py:185` `test_multi_turn_supervises_every_assistant_turn` | Delete | Strictly weaker than `test_supervised_tokens_are_exactly_assistant_content_plus_eot[multi_turn-*]` (`:156`): same path, same fixtures, superset parameterization; the exact-equality assertion implies all three of this test's assertions (verified empirically). |
| `tests/training/test_sft.py:219` `test_rendered_segments_concatenate_to_golden` | Delete or mark diagnostic | A 20-mutation harness found zero mutations this test catches that the round-trip golden test (`:211`) does not; its only value is failure localization. |
| `tests/training/test_replay.py:144` `test_one_epoch_covers_every_window_once` | Delete | Character-identical assertions to `test_loads_all_windows_when_budget_fits` (`:54-58`) minus one implied length check; same fixture, same seed. |
| `tests/training/test_sft_entry.py:153` `test_no_cross_example_loss_bleed` | Fold or rewrite multi-sequence | On a single packed sequence the invariant holds by construction given `test_labels_preserved_verbatim`; a flush-path mask bug passes all of TestPacking today (mutation-verified) — the rewrite over a multi-sequence pack targets a real gap. |
| `tests/training/test_sft_entry.py:85` `test_gemma4_is_supported` | Delete | Membership assertion on `HF_SUPPORTED_MODEL_TYPES` catches nothing the neighboring mapping tests do not (simulated removal fails them). |
| `tests/interp/test_probes.py:27` `test_order_independent` | Rewrite | First assertion compares a call to an identical call (cannot fail); second is subsumed by `test_proportions`. Nothing permutes order — shuffle the dataset and assert identical per-text assignments. |
| `tests/interp/test_harvest.py:257` `hook.complete` assertion | Replace | Vacuously true: nothing in the codebase ever creates a file of that name (grep plus git history). Assert the kept capture's cache.bin survives instead. |
| `tests/training/test_pipeline.py:206` empty-dir branch | Fix setup | `tmp_path/"empty"` is never created, so the test passes via the missing-path `open()` error while the empty-directory guard (`training/data/prepare.py:118`) stays unexecuted (coverage-confirmed). `mkdir()` first, match `'No .jsonl'`. |
| `tests/training/test_contamination.py:86` `test_index_hash_reflects_benchmark_list_changes` | Rewrite | Vacuous: `index_hash` hashes a manifest embedding a microsecond timestamp, so two builds always differ (reproduced with byte-identical inputs). Assert `ngrams_xxhash64` changes instead, or make `index_hash` content-stable. |
| `tests/training/test_serving.py:14` `test_registry_matches_compose_definitions` | Rewrite | Re-declares the constants by hand and never reads docker-compose.yaml, so the drift it is named for passes silently; also omits llama-cpu entirely. Parse compose (pyyaml is available) and compare all three services. |
| `tests/training/test_distill.py:198` | Delete line | `torch_loss = pytest.importorskip` aliases the function without calling it — a no-op editing remnant; real gating is the autouse fixture at `:205-208`. |

Two further redundancy candidates were investigated and rejected: the hardcoded family
lists in `test_sft.py` are backstopped by `test_peft_utils.py:38`'s exact-equality
registry assertion (a new family fails loudly), and `test_clean_absolute_path` in
`test_worktree_make.py` pins a genuinely distinct input class (mutation-verified).

## Skip architecture: silent loss of sandbox signal

The suite's biggest structural weakness is module-level `pytest.importorskip`, which
throws away sandbox-runnable tests wholesale:

- `tests/interp/test_steering.py:22` gated all 26 steering tests on gguf, though only
  the export tests need it. Wiring, extract-input, and judge-parsing tests run fine
  without it. This single gate is why steering coverage previously read 25-31 percent;
  with gguf present, `interp/steering/wiring.py` measures 92.5 percent. Scope the skip
  to the export tests.
- `tests/training/test_schedules.py:12` skips `TestWsdMultiplier` — pure math, and the
  P2-D3 acceptance's named sandbox test — although `training/schedules.py` imports
  torch-free. Split the curve tests into a torch-free module.
- `tests/training/test_checkpoint.py:13` skips retention/`resolve_resume` tests that
  need only fabricated step directories (`test_watchdog.py:178-191` already does this
  torch-free); `resolve_resume` measures zero sandbox coverage as a result.
- `tests/training/test_loader.py:12` hides two torch-free `resume_index` tests
  (verified to pass without torch when moved above the gate).

Convention to adopt: module-level gates only when every test in the file genuinely
needs the dependency; otherwise per-class or fixture gates, keeping pure-logic tests
running where the suite actually runs.

## Dead code

Every item verified to have zero live references, including dynamic dispatch, Makefile,
docker, docs, and YAML:

| Location | Item | Note |
|----------|------|------|
| `interp/sae/registry.py:22` | `SaeRegistryError` | Never raised; `load_sae_run` raises `ManifestError`. Its only catch (`scripts/interp/eval_sae.py:76`) is unreachable. The mirrored probes registry does raise its error — the SAE registry omitted the wiring (see duplication). |
| `interp/probes/evaluate.py:35` | `EvaluateError` | Never raised; the docstring's baseline-failure condition was never implemented. |
| `training/eval/judge.py:35` | `JudgeError` | Never raised, caught, or imported beyond its `__init__` re-export; judge.py contains no raise statement. |
| `interp/organisms/generate.py:32` | `QA_TEMPLATES` | Names a template file that does not exist; design superseded by teacher-generated QA. All sibling constants are consumed. |
| `interp/probes/monitor.py:30` | `DEFAULT_LOG_ROOT` | Never read; the CLI hardcodes its own default, duplicating the path literal. |
| `interp/steering/extract.py:44` | `DirectionSet.layer_count` | Never called; the exporter recomputes `max(directions)` inline. |
| `interp/steering/scorecard.py:75` | `ScorecardConfig.extra` | Never populated or read; report writer enumerates fields explicitly. |
| `training/distill.py:271-303` | `DistillTrainer.last_ce`/`last_kl` | Write-only. Either log train/ce and train/kl through the tracking callback (useful distillation diagnostics) or delete. |
| `training/sft.py:71` | `SftResult.packed_examples` | Write-only; both CLI outputs enumerate fields without it. Surface it or delete. |
| `training/loop.py:74` | `TrainResult.data_wait_ms_mean` | Computed but never emitted, although `docs/training/heterogeneous_compute.md:99` documents it as the D9 acceptance metric carried by the summary. Surface it in `cmd_train` output (preferred) or correct the doc. |
| `training/calculator/calibrate.py:38` | `build_model` alias | Zero consumers; the module itself calls `build_scratch_model`. |
| `scripts/eval/eval_ft.py:51` | `DEFAULT_TASKS` | Never read; the real default is `training/eval/lmeval.py:19`. |
| `training/data/contamination.py:168` | `ContaminationIndex.hash` | Write-only attribute recomputed per pipeline worker. |
| `scripts/agents/preflight-scan.sh:30,25` | `print_warn`, `BLUE` | Defined, never used; the script already trims the canonical helper set. |
| `package.json`, `package-lock.json` (root) | Empty `{}` manifests | Stray npm artifacts from f808ed0; the real manifest is `.opencode/package.json`. Delete after a host-side check that opencode plugin resolution does not need a root package.json. |
| `interp/output.py` | Printer re-export shim | Not yet deletable: still imported by `scripts/interp/train_sae.py:16` and `eval_sae.py:19` — see the CLI-migration item below. |

Also stale, doc-side: three files point at `docker/llama-cpu/hugepage_mmap_wrapper.cpp`,
which does not exist (`AGENTS.md:62`, `.styles/CPP_STYLE.md:199`,
`hugepages/docs/hugepages-explicit.md:231`); the real interceptor is
`hugepages/src/memory/hugepage_wrapper.cpp`.

## Duplicated code

### Unfinished consolidation from PR #48

The unified-CLI refactor stopped one step short in three places:

1. `scripts/interp/train_sae.py` and `eval_sae.py` are the only interp entry points
   left outside `python -m interp` (P4-D3 landed hours before the refactor). They keep
   the pre-refactor `sys.path.insert(0, ".")` pattern, are the sole importers of the
   `interp/output.py` shim, and their argparse wiring has no tests. Fold them in as
   `sae-train`/`sae-eval` subcommands, repoint `Makefile:289,295`, add the two lines to
   `test_interp_cli.py` CASES, then delete the shim.
2. Four script-local `Printer` copies survive (~148 lines): `scripts/eval/eval_ft.py:55`,
   `scripts/eval/serve_eval.py:48` (both already sys.path-bootstrap the repo root, so
   `from common.output import Printer` costs nothing), plus the standalone
   `scripts/eval/promote_scorecard.py:34` and `scripts/utils/check_py_deps_install.py:59`
   (stdlib-only by design — consolidate with the two-line bootstrap or document the
   exemption).
3. `training/cli.py` mixes 9 inline `cmd_*` handlers with 9 modular
   `training/commands/` registrations, while `interp/cli.py` is fully modular. Low
   priority, mechanical.

### Helper duplication in source

| Helper | Copies | Recommendation |
|--------|--------|----------------|
| `sha256_file` | 3 byte-identical: `training/serving.py:90`, `training/export/manifest.py:19`, `scripts/eval/promote_scorecard.py:61` | `common/hashing.py`; keep the xxh3/xxh64 variants' recorded algorithms as thin wrappers (`interp/manifest.py:44`, `training/data/shards.py:41`). |
| git-SHA helper | 8 variants with drifted failure contracts ('unknown' vs None vs raise; cwd-based vs repo-root-pinned): `training/loop.py:87`, `training/sft_callbacks.py:59`, `training/data/tokenizer.py:162`, `training/data/prepare.py:156`, `training/data/teacher.py:231`, `interp/manifest.py:29`, `training/eval/scorecard.py:29`, `scripts/eval/promote_scorecard.py:69` | `common/provenance.py` with one repo-root-pinned `git_sha()`; decide deliberately whether promote_scorecard keeps its hard-fail. |
| `quant_from_name` | 3 byte-identical: `training/eval/runner.py:34`, `training/data/teacher.py:242`, `scripts/eval/serve_eval.py:90` | Single canonical copy; this function feeds the quantization-confound bookkeeping, so silent drift between copies corrupts the guard. Fold the case-normalization fix (B17) into the consolidation. |
| `positive_int` | 4 byte-identical argparse helpers in `training/commands/{prepare_data,prepare_corpus,prepare_sft,train_tokenizer}.py` | One shared definition plus one test (none of the four copies is tested). |
| OpenAI-compatible chat client | 2 divergent: `training/serving.py:172` (urllib, transient/permanent error taxonomy) vs `interp/steering/scorecard.py:137` (requests, no taxonomy) — and `interp/commands/probe_eval.py:32` already imports the scorecard's client, treating it as shared infrastructure | Move `ServiceSpec`/`SERVICES`/`chat_completion`/`ensure_training_idle` to `common/serving.py` (training re-exports); port the interp callers. This is also where the B3 lock-path bug and the failure-2 GPU-consumer flag land naturally. |
| Registry pair | `interp/probes/registry.py` vs `interp/sae/registry.py`: identical listing logic, drifted error contracts (probes wraps into `RegistryError` and hash-verifies the artifact; SAE lets `ManifestError` escape and verifies no hash) | Shared `list_artifact_dirs`/`require_identity` in `interp/manifest.py`; align exception wrapping. |
| Smoke tests | `interp/commands/smoke_test.py` vs `training/commands/smoke_test.py`: 42 percent line-identical (measured), byte-identical `_pkg_version`/`_driver_version`/`_needs_cuda` | Extract a `common/smoke.py` step-runner; each tier keeps constants and step bodies. |
| Manifest writers | 3 durability contracts: atomic (`interp/manifest.py:63`), atomic+fsync (`training/data/shards.py:128`), plain non-atomic (`training/export/manifest.py:104`) — a crash mid-write leaves a truncated export manifest beside a valid GGUF | `write_json_atomic` in common/; the export writer becoming atomic is a strict improvement. |
| Steering env contract | `interp/steering/wiring.py` (test-only) vs `Makefile:223-234` (production) implement the same CONTROL_VECTOR env/path mapping independently | Have make shell out to a small `python -m interp steer-env` helper, or cross-link both sites; do not delete `steering_env` without deciding which side owns the contract. |
| `tokenizer_fingerprint` | Defined in `interp/probes/capture.py:49` but consumed by probes, SAEs, and harvests; `interp/harvest.py:206` imports across subpackages for it | Move to `interp/manifest.py` beside `text_sha256`. |

### Test-side duplication

The largest cluster: the ~12-line synthetic shard-dir builder (ShardWriter plus
11-key manifest) is copy-pasted in eight files (`test_shards.py:20`,
`test_loader.py:38`, `test_staging.py:19`, `test_replay.py:30`, `test_mixture.py:17`,
`test_loop.py:38`, `test_schedules.py:80`, `test_watchdog.py:206`). A
`make_shard_dir(tmp_path, tokens, *, shard_tokens, dtype, tokenizer_hash)` fixture in
`tests/training/conftest.py` collapses all eight — and parameterizing dtype makes the
missing uint32 test (below) trivial. Also confirmed: `make_source` byte-identical in
test_mixture/test_replay; the tiny BPE tokenizer builder written out three times; the
benchmarks-yaml builder duplicated between test_pipeline and test_contamination; the
autouse `TRAIN_LOCK_PATH` fixture duplicated in test_harvest/test_sae instead of living
in `tests/interp/conftest.py`; `REPO_ROOT` re-derived in 15 files (a
`tests/training/support.py` for REPO_ROOT/REFERENCE_DIR/`reference_recipe()` fixes the
module-constant-cannot-use-fixture problem); tests/agents lacks a conftest while
test_sandbox_seed and test_worktree_make duplicate a throwaway-repo fixture
byte-for-byte; and `tests/interp/conftest.py:35` seeds fake captures from salted
`hash()`, so its "bit-identical" docstring holds only within one process — seed from
hashlib before any golden-file comparison is added.

## Coverage gaps and proposed tests

Measured coverage is 62.2 percent; 43 files sit below 75 percent. Three structural
causes dominate, and fixing them moves coverage more than any individual test:

1. **Module-level torch gates skip existing pure tests** (section above).
2. **Both CLIs are parse-tested only.** No test executes any handler body:
   `training/cli.py` is at 22.2 percent over a 92-94 percent covered calculator, and
   seven core subcommands (train, train-sft, train-distill, distill-cache, export,
   eval-holdout, eval-lmeval) have no tests at any level — no parse binding, no
   exit-code contract, no `--json` purity. The interp CLI is the same: zero handler
   coverage across all ten subcommands, including a live defect the first CLI test
   would catch (`interp/commands/steer_scorecard.py:60` builds
   `tuple(float(s) for s in args.scales.split(','))` outside its try block, so a
   malformed `--scales` escapes as a raw traceback instead of a styled exit 2).
3. **Import-before-validate defeats designed refusals** (failures section): fixing
   B-item import ordering in checkpoint.py, distill.py, and holdout.py makes roughly a
   dozen documented refusal messages sandbox-testable at once.

### Highest-value new tests (sandbox-runnable unless noted)

Family and policy contracts:

- **uint32 shard round trip** (`tests/training/test_shards.py`): no test ever writes or
  reads uint32 shards, yet Gemma 4's 262,144-token vocab makes uint32 the production
  path for every real shard. Related: `ShardWriter.write_tokens` silently wraps
  out-of-range ids when handed an int64 ndarray (verified) — add a range guard and pin
  both list and ndarray behavior.
- **qwen3 calculator mapping** (`tests/training/test_calculator.py`): the fallback
  family's `spec_from_hf_config` path is exercised by no test, and Qwen3 checkpoints
  carry an explicit `head_dim` decoupled from hidden//heads — exactly the field the F3
  fix introduced. Move `TestCalculatorGemma4` out of `test_sft_entry.py` into
  `test_calculator.py` at the same time (the file split is how this gap went unnoticed).
- **Gemma 4 MoE refusal** (with B10) and **chat-template-version refusal** (with B7).
- **interp recipe validation**: none of the eight committed recipes under
  `training/recipes/interp/` — including the only executable Qwen3-fallback artifacts
  in the interp tier — is loaded by any test, unlike organism recipes
  (`test_organisms.py:410`). One parametrized load-and-validate test.
- **`resolve_d_model` text_config descent** (`interp/harvest.py:144`): the Gemma 4
  composite-config contract, testable with a stubbed `transformers.AutoConfig`; every
  harvest test currently bypasses it via the d_model override. Pair with the B15
  hook-path test using a fake nested module tree.
- **`tokenizer_fingerprint` tracks template changes** (`interp/probes/capture.py:49`):
  the W2 guard that makes a Gemma 4 vs Qwen3 template mismatch detectable is never
  directly tested.
- **Multi-turn golden renderings**: `GOLDEN_RENDERED` has single_turn and system
  fixtures per family but no multi_turn golden for either family.

Serving and eval infrastructure (all currently at or near zero coverage):

- `chat_completion` error taxonomy (5xx transient vs 4xx fatal vs malformed body) and
  `wait_healthy` timeout — the classification every teacher/eval retry loop keys on
  (`training/serving.py:154-205`, mocked urlopen).
- `generate_responses` transient-retry healing and `run_judgments` error-then-resume
  re-judging (`training/eval/generate.py:87`, `training/eval/runner.py:151`).
- `extract_results` including the comma-metric stderr key composition and both refusals
  (`training/eval/lmeval.py:49`); the lm-eval version-mismatch guard via a stubbed
  `lm_eval` module (`:84`).
- `resolve_model` forms (hub id, checkpoint dir, adapter) in the 0 percent
  `training/eval/holdout.py`, whose docstring explicitly promises unit-testable pure
  helpers.

Refusal and resilience paths with designed messages but no coverage: the logproxy proxy
route (streamed relay, 502-plus-record, passthrough — the W3 digest calls this
sandbox-testable with a fake upstream; the module is also invisible to coverage because
it loads via importlib from docker/), tracking's fail-soft sink eviction
(`training/tracking.py:77-89`), the watchdog sampler thread and gpu-throttle/ram-floor
re-arm paths, `run_lock` corrupt-lock staleness, staging copy-corruption OSError,
`prepare()` extra_manifest collision guard, teacher-config validation errors, six
unlisted `validate()` cross-field branches (`training/recipes/validate.py`), probe/SAE
registry error paths, steering scorecard iteration-count and judge-range refusals, and
the W1 entrypoint byte-identity check (fake llama-server via the existing `fake_bin`
fixture, asserting no control-vector flags leak when unset).

Agents and utils: guard.sh has no test for the Bash models-dir branch at all, nor for
lowercase tool names (the OpenCode normalization path); preflight-scan has no
clean-directory pass case and most inventory vectors are unplanted;
check-agent-versions.sh is the only agents script with zero tests — and probing it
found two of the bugs above. `common/output.py` itself — the shared Printer
implementing the style contract for every CLI — has no test.

Deliberate non-goals: the two smoke_test.py modules (9-12 percent) are host-side by
design; rather than chase unit coverage, add
`[tool.coverage.run] omit = ["*/__main__.py", "*/commands/smoke_test.py"]` to
pyproject.toml so the denominator reflects sandbox-testable code, and keep the existing
`--build` degrade-gracefully test proposals modest.

## Gemma 4 and Qwen3 readiness (July 2026)

Facts verified during this audit (sources: `configuration_gemma4.py` read from
transformers main; the official chat template mirror read from the reference gist;
huggingface.co/blog/gemma4 and the two ai.google.dev pages were unreachable from this
container and their content was reconstructed from search snippets — marked SEARCH):

- **Gemma 4** (April 2026; transformers >= 5.5.0, repo pins 5.13.1): sizes E2B, E4B,
  12B, 26B-A4B (MoE), 31B. Text config: vocab 262,144; explicit `head_dim`; 5:1
  sliding:full attention cycle; per-layer embeddings (the E-model mechanism);
  `enable_moe_block` flag on the same gemma4/gemma4_text model_types. Chat format:
  `<|turn>role\n...<turn|>\n`, BOS once, native system turn, assistant role `model`.
  Thinking: opt-in `<|think|>` in the system turn, reasoning in
  `<|channel>thought...<channel|>` sections, thought channels stripped from history by
  the official template (SEARCH for sizes/context; format READ).
- **Qwen3** (fallback): ChatML framing, no BOS, dense decoder with explicit `head_dim`.
  Hybrid checkpoints — including the repo-pinned Qwen/Qwen3-1.7B — **think by
  default** (`enable_thinking=True` in `apply_chat_template`); the 2507 refresh split
  into Instruct and Thinking-only variants (SEARCH, medium-high confidence).

The repo is already strongly Gemma-4-first and the core contracts check out: the family
registry (`training/families.py`) is the single source of truth and fails loudly; the
v2 chat template spec matches the official template read during this audit (framing,
`model` role, BOS-once, trimming); the calculator prices PLE and descends
`text_config`; offload freeze patterns cover the PLE table; all six reference recipes
and all three organism recipes target gemma-4-E2B/E4B.

Gaps against the "Gemma 4 primary, Qwen3 fallback" policy, beyond bugs B7/B9/B10
already listed:

1. **The Qwen3 fallback is registered but not operational.** No reference fine-tune
   recipe, no dataset-prep example, no calculator test exercises it; the registry's
   stated purpose — "a Gemma 4 library-support failure never blocks the phase" — has no
   executable path to switch to. Add an `sft-qlora-qwen3` reference recipe and include
   it in `test_recipes.py`'s enumerated set.
2. **No thinking-trace handling exists anywhere.** Beyond the B9 eval default, nothing
   at data-prep time validates that assistant turns are free of `<|channel>thought`
   (Gemma 4) or `<think>` (Qwen3) markers; a trace-contaminated source dataset would be
   supervised on raw reasoning markers with no test or guard. Decide and document the
   empty-think-block convention for Qwen3 SFT while adding the guard.
3. **Interp hook paths assume the flat text-model layout.** `hook_name` hardcodes
   `model.layers.{n}` (`interp/harvest.py:52`) and the nnsight capture hardcodes
   `model.model.layers[layer]` (`interp/probes/capture.py:94`), while only
   `resolve_d_model` got composite-config descent. A multimodal-composite Gemma 4
   checkpoint would make every hook miss, and B15 turns that into a raw traceback. Run
   `make harvest-check-hooks` against google/gemma-4-E2B host-side before any GPU
   capture budget is spent.
4. Minor: `HF_SUPPORTED_MODEL_TYPES` still lists qwen2/gemma/gemma2/gemma3/mistral —
   correct for a measurement tool but easily misread as the policy family list; one
   comment distinguishing the two prevents confusion.
   `docs/training/d7_train_serve_investigation.md:37` still scopes GGUF export to
   "Llama/Qwen/SmolLM families" while the live code message says "Gemma 4 primary,
   Qwen3 fallback" — the doc is the bug per repo rules.

## Process recommendations

1. **Pin the sandbox dependency set.** Add a `sandbox` dependency group (numpy,
   omegaconf, pyyaml, xxhash, tokenizers, scikit-learn, hydra-core, fastapi, httpx,
   gguf, pytest plugins) so a fresh sandbox can `uv sync --group sandbox --group dev`
   and run the full 716-test extent instead of 137. A SessionStart hook for remote
   sessions could install it automatically.
2. **Add the coverage config** (`omit` for `__main__.py` and smoke tests) so the
   62.2 percent headline stops undercounting by design.
3. **Adopt the skip-granularity convention** from the skip-architecture section.
4. **Consolidate test scaffolding** (conftest fixtures and the support module from the
   duplication section) before writing the new tests above, so they land on shared
   fixtures rather than a ninth shard-builder copy.
5. **Env template hygiene**: `env.example` documents `LLAMA_MODEL` ("takes precedence")
   which nothing reads — compose reads `LLAMA_CPU_MODEL` — and omits the consumed
   `TRAIN_STAGING_GB`, `TRAIN_LD_PRELOAD`, and `LOGPROXY_TARGET_*` variables. Fix
   alongside B2. Add the ##@ Training targets to `.PHONY` (currently none are listed).
   Stale doc lines recorded during the audit: `docs/interpretability/writeback.md:7`
   (CLIs moved to `interp/commands/`), `docs/unified_cli_investigation.md:28-49`
   (describes the migration as not started), `docs/training/data_pipeline.md:86`
   ("until the D3 tokenizer trainer lands" — it landed),
   `docs/sandbox/docker_compose_overview.md:499-541` (documents the dead
   `LLAMA_MODEL`), plus the hugepage wrapper path above.

## Prioritized action list

P0 — broken now, fix first:

1. B1 export container missing `common/` (host-verify after fix).
2. B2 llama-cpu entrypoint variable mismatch (host-verify after fix).
3. Failure 2 decision: llama-cpu in the training preflight (option 1 or 2), then fix
   the stale test accordingly.
4. B3 steering scorecard GPU lock path.
5. B4/B5 guard.sh fail-open and downloader gap; B6 runpod float parsing.

P1 — correctness guards and their tests:

6. Import-order fixes (checkpoint.py, distill.py, holdout.py) — unblocks the failing
   test plus a dozen refusal tests.
7. B7 chat-template-version enforcement; B8 distill base identity; B10 MoE guard.
8. B9 family-aware thinking_mode plus the Qwen3 thinking-trace decision.
9. Fix the two env-coupled test_check_py_deps tests and the vacuous/misfiring tests in
   the redundancy table.
10. B11-B22 in descending order.

P2 — structure:

11. Skip-granularity fixes; sandbox dependency group; coverage omit config.
12. Test scaffolding consolidation, then the highest-value new tests (family/policy
    contracts first, serving/eval taxonomy second, refusal paths third).
13. SAE scripts into the unified CLI; Printer and helper consolidation into common/.

P3 — hygiene: dead-code deletions, doc corrections, `.PHONY`, env templates,
`training/cli.py` modularization.

## Appendix: modules below 75 percent coverage

Sandbox measurement with the full dependency set installed; smoke tests and
`__main__.py` stubs included (see the omit recommendation).

| Coverage | Stmts | Module | Dominant cause |
|----------|-------|--------|----------------|
| 0.0% | 63 | training/calculator/calibrate.py | GPU-only by design; pure `write_record`/`as_dict` untested |
| 0.0% | 54 | training/eval/holdout.py | torch-import-before-guard; pure `resolve_model` untested |
| 9.3% | 151 | training/commands/smoke_test.py | host-side by design (omit) |
| 11.6% | 155 | interp/commands/smoke_test.py | host-side by design (omit) |
| 16.7% | 66 | training/tracking.py | fail-soft contract untested |
| 17.4% | 46 | training/model.py | GPU builders; untested |
| 19.5% | 267 | training/distill.py | torch-first guards; untested pure preflights |
| 20.7% | 121 | training/commands/train_tokenizer.py | cmd() body never executed |
| 22.2% | 423 | training/cli.py | parse-tested only |
| 22.4% | 201 | interp/commands/organism.py | serving-dependent, plus untested sandbox paths |
| 23.7% | 38 | training/schedules.py | pure math behind torch gate |
| 23.9% | 226 | training/loop.py | torch loop; pure helpers untested |
| 31.2% | 32 | interp/commands/harvest.py | handler untested |
| 31.7% | 167 | training/checkpoint.py | torch gate; pure retention/resolve untested |
| 32.1% | 81 | training/commands/generate_teacher.py | restore-on-failure contract untested |
| 33.3% | 33 | training/commands/validate_recipe.py | cmd() body never executed |
| 34.3% | 35 | interp/commands/probe_eval.py | handler untested |
| 34.5% | 29 | interp/commands/interp_gc.py | handler untested |
| 35.8% | 95 | training/data/loader.py | torch gate hides numpy-only tests |
| 36.7% | 60 | training/commands/prepare_sft.py | cmd() body never executed |
| 37.0% | 27 | training/commands/build_contamination.py | cmd() body never executed |
| 38.2% | 191 | training/data/pipeline.py | engine-free funnel helpers only run under datatrove gate |
| 39.4% | 33 | common/output.py | shared Printer untested |
| 41.3% | 92 | interp/sae/evaluate.py | GPU eval; pure parts untested |
| 41.5% | 65 | training/commands/prepare_corpus.py | cmd() body never executed |
| 42.5% | 40 | interp/commands/steer_extract.py | handler untested |
| 42.5% | 40 | training/eval/lmeval.py | pure extract_results untested |
| 45.8% | 24 | interp/commands/probe_train.py | handler untested |
| 46.0% | 50 | training/commands/prepare_data.py | cmd() body never executed |
| 46.5% | 43 | training/commands/make_replay_synthetic.py | cmd() body never executed |
| 47.6% | 63 | interp/steering/extract.py | torch/repeng extract paths host-side; error paths untested |
| 48.5% | 33 | interp/commands/probe_monitor.py | handler untested |
| 50.0% | 36 | interp/commands/steer_scorecard.py | handler untested (has the --scales traceback defect) |
| 50.0% | 32 | training/offload.py | torch gate; froze-everything refusal untested anywhere |
| 55.8% | 206 | training/sft.py | trainer body torch-gated; probe/build_dataset untested |
| 58.3% | 24 | interp/commands/probe_capture.py | handler untested |
| 59.8% | 132 | training/sft_callbacks.py | callbacks need transformers stub |
| 63.3% | 196 | training/watchdog.py | sampler thread and two signal paths untested |
| 66.3% | 98 | training/serving.py | chat_completion/wait_healthy at 0% |
| 69.0% | 87 | interp/probes/capture.py | tokenizer_fingerprint and loader errors untested |
| 73.4% | 192 | interp/harvest.py | resolve_d_model, check_hook_dims untested |
| 74.5% | 51 | training/peft_utils.py | Params4bit doubling contract untested |

Numbers above reflect the sandbox with gguf/fastapi installed; the previously reported
25-31 percent figures for `interp/steering/wiring.py`, `export.py`, and `scorecard.py`
were artifacts of the module-level gguf skip (wiring measures 92.5 percent with the
skip scoped correctly).
