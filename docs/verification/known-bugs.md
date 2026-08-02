# Known bugs by runbook step

Every defect the WI-14 capstone review and its Opus re-verification confirmed, indexed so the operator running the [hardware verification runbook](README.md) knows which known bugs to expect at each step. The code passed its sandbox unit tests; these were latent defects in paths those tests never exercised. Of the 51 originally ledgered, 36 pure-Python defects were since fixed with regression tests in the sandbox pre-fix pass (see Resolved below); the 15 still open are GPU-runtime, pinned-library, or design-decision defects the sandbox cannot observe. Fix those against the real traceback as each step reaches them.

## How to read this

- **KB-nn** is a stable identifier; the runbook steps reference these IDs. IDs are ordered by the first campaign step where the bug appears.
- **Step(s)** are the runbook entries (README phase files) where the bug manifests; a bug listed at more than one step recurs at each.
- Each was confirmed by an independent three-reviewer adversarial panel (reviewers instructed to refute by default; a bug is listed only when at least two of three upheld it). Confidence is the upheld count.
- These are open unless a later commit resolves them; when one is fixed, move its row to Resolved rather than deleting it, so the IDs stay stable.

## Open known bugs (16)

| ID | Step(s) | Location | Category | Confidence | Defect |
|----|---------|----------|----------|------------|--------|
| KB-52 | P0.1 | `training/data/prepare.py:210` | teardown | 3/3 | Streaming prepare-data crashes at interpreter finalization (PyGILState_Release fatal error) after all shards and the manifest are flushed: a datasets/aiohttp background download thread is still alive at exit, so the process returns nonzero on an otherwise successful run (observed on the workstation 2026-08-01, P0.1) |
| KB-03 | P0.3 | `training/calculator/calibrate.py:89` | cross-module-contract | 2/3 | calibrate() always builds a scratch LlamaConfig model, so calibrating any model.hf (or PEFT) recipe measures a different model than the one the estimate prices. Fix landed 2026-08-01, so the location above is the pre-fix site (calibrate now builds through the shared `build_model` + PEFT wrap in `_build_calibration_model`, `calibrate.py:121`; `zero_offload` and online-logit-distill recipes refused as uncalibratable); pending workstation confirmation via the P1 QLoRA calibration runs (`sft-qlora-e2b`, `sft-qlora-gemma-e4b`) |
| KB-17 | P1.6 | `training/tune/distill.py:541` | correctness | 3/3 | Cached-mode distillation never checks logit-softcap parity anywhere, despite the documented refuse-at-startup contract |
| KB-18 | P1.9 | `pyproject.toml:161` | pinned-api-contract | 3/3 | lm-eval==0.4.12 is pinned without the deps its own pinned task set requires: ifeval needs langdetect/immutabledict ([ifeval] extra) and tinyMMLU needs tinyBenchmarks — both absent from pyproject and uv.lock |
| KB-19 | P1.9, P1.10 | `training/eval/tasks.yaml:15` | pinned-api-contract | 3/3 | tasks.yaml pins bare metric keys for ifeval/tinyMMLU/hellaswag, but lm-eval 0.4.12 keys every aggregate result as "<metric>,<filter>" — extract_results raises after the full harness run completes |
| KB-22 | P1.10 | `training/eval/generate.py:101` | resume-integrity | 3/3 | Response-record resume matches on prompt id only, so a prompt whose text changed between runs reuses the stale response while the scorecard stamps the new prompt-set hash |
| KB-26 | P2.2 | `training/data/pipeline.py:224` | correctness | 3/3 | Stale datatrove completion markers make prepare_corpus reruns silently reuse old stage output while the manifest records the new config and contamination index |
| KB-28 | P2.6 | `training/runtime/loop.py:241` | cross-module-contract | 3/3 | memory.offload=zero_offload with a non-cpu_adam optimizer silently disables offload while the calculator and validation treat the state as offloaded |
| KB-29 | P2.6 | `training/runtime/loop.py:258` | cross-module-contract | 3/3 | Plain loop silently ignores model.trainable_filter unless memory.offload is zero_offload, training frozen-by-policy parameters |
| KB-30 | P2.6 | `training/runtime/offload.py:119` | correctness | 3/3 | build_ds_config's steps_per_print: 0 makes DeepSpeed 0.19.2 raise ZeroDivisionError on the first optimizer step, crashing every real ZeRO-Offload run |
| KB-31 | I.3 | `interp/steering/extract.py:106` | correctness | 3/3 | extract_repeng never passes its computed layer list to ControlVector.train, so layer_range is silently ignored and the exported vector/manifest contain directions for every layer |
| KB-32 | I.3 | `interp/steering/extract.py:78` | pinned-api | 3/3 | Pinned repeng 0.4.0 cannot be imported under pinned numpy 2.4.6: np.float_ in eagerly evaluated annotations raises AttributeError, killing every repeng extraction |
| KB-37 | I.5 | `interp/probes/capture.py:93` | correctness | 3/3 | capture_activations indexes the batch dimension: decoder blocks return a bare tensor under pinned transformers 5.13.1, so block.output[0] takes the first sequence, and every real multi-sequence capture crashes (or silently corrupts) at the last-token gather |
| KB-40 | I.7 | `interp/probes/monitor.py:211` | cross-module-contract | 3/3 | Monitor replay always loads the hub 'main' revision: run_monitor never plumbs a revision to score_texts, and the probe manifest drops model_revision |
| KB-45 | I.8 | `interp/organisms/verify.py:221` | input-validation | 3/3 | locality_check's passed verdict ignores unmatched metrics despite its documented contract |
| KB-50 | I.11 | `interp/sae/train.py:223` | correctness | 3/3 | capture_manifest_xxhash is invalidated by mark_validated within the same run: run_sae_training hashes the capture manifest, then mutates it |

## Resolved during the hardware campaign (4)

Found by the campaign itself (not the WI-14 panel), fixed against the real
workstation traceback per the runbook protocol, and confirmed resolved by the
P0.3 calibrate re-runs on 2026-08-01.

| ID | Step(s) | Location | Category | Confidence | Defect |
|----|---------|----------|----------|------------|--------|
| KB-56 | P0.3 | `training/calculator/solve.py:100` | correctness | 3/3 | allocated_total summed the full checkpointed activation inventory as if simultaneously live, but the allocator peak is temporal (gradients are ~0 at head backward; head tensors are freed before gradients fully materialize) — pretrain-1b estimated 11.42 GB vs measured 7.00 GB, +63.2 percent (workstation 2026-08-01); fix models the checkpointed total as weights + optimizer + max(late-backward, head-backward), confirmed by the 2026-08-01 workstation re-run: +2.1 percent allocated error |
| KB-55 | P0.3 | `training/calculator/components.py:321` | cross-module-contract | 3/3 | grad_bytes priced gradients at precision.grad_dtype (fp32, 4.00 GB at 1B), but grad_dtype had no consumer outside the calculator — no trainer implements fp32 gradient accumulation, so a bf16 model carries bf16 gradients (2.00 GB); fix prices gradients in the parameter dtype (peft compute_dtype under PEFT) and removes the dead grad_dtype field, confirmed by the 2026-08-01 workstation re-run: +2.1 percent allocated error |
| KB-54 | P0.3 | `training/cli.py:195` | acceptance-criteria | 3/3 | The calibrate allocated-error check was purely relative (10 percent), but max_memory_allocated includes ~65 MB of fixed CUDA library workspaces (cuBLAS) outside the component model, so the bar could never pass on the tiny-e2e reference recipe (-93.1 percent observed on the workstation 2026-08-01); fix adds a 128 MiB absolute floor (ALLOCATED_FLOOR_BYTES) alongside the relative bar, confirmed by the 2026-08-01 workstation re-run: tiny-e2e gap 0.06 GB within the floor |
| KB-53 | P0.3 | `training/calculator/calibrate.py:87` | initialization | 3/3 | calibrate() calls torch.cuda.reset_peak_memory_stats before any CUDA context exists; the memory-stats API does not lazy-init the context, so a fresh container always fails with "Invalid device argument" (isolated by a minimal probe on the workstation 2026-08-01; fix adds torch.cuda.init() before the reset, confirmed by the 2026-08-01 workstation re-run: calibrate loop ran to completion) |

## Resolved in the sandbox pre-fix pass (36)

Fixed with per-bug regression tests in PR #65 (`fix the 36 sandbox-provable known bugs before the hardware campaign`) before the campaign reached these steps. Every fix is a pure-Python path provable in the sandbox; the 15 bugs still open above are pinned-library, GPU-runtime, or design-decision defects that must be fixed against the real traceback per the runbook protocol. Each regression test was demonstrated failing on the pre-fix code.

| ID | Step(s) | Location | Category | Confidence | Defect |
|----|---------|----------|----------|------------|--------|
| KB-01 | P0.1, P2.2 | `training/data/prepare.py:180` | correctness | 3/3 | _map_encode reuses the module-level _WORKER_TOKENIZER without checking spec, silently tokenizing with the wrong tokenizer while the manifest records the right one |
| KB-02 | P0.2 | `training/data/tokenizer.py:215` | manifest-accuracy | 3/3 | Tokenizer artifact manifest records the requested vocab_size, not the trained tokenizer's actual vocabulary size |
| KB-04 | P0.4, P0.3 | `training/recipes/validate.py:177` | cross-module-contract | 3/3 | precision.grad_dtype is never validated; values outside DTYPE_BYTES pass validate() and crash the calculator with a raw KeyError (the field, its check, and its regression test were later removed entirely with the KB-55 fix — grad_dtype was never consumed outside the calculator) |
| KB-05 | P2.3, P0.4 | `training/recipes/validate.py:525` | correctness | 3/3 | validate() escapes with raw JSONDecodeError/AttributeError instead of RecipeValidationError when a shard manifest.json is corrupt or not a JSON object |
| KB-06 | P0.6 | `training/runtime/checkpoint.py:136` | kill-safety | 3/3 | save_checkpoint deletes an existing published step dir before renaming its replacement in, so a SIGKILL in between destroys the checkpoint `latest` points to and leaves `latest` dangling |
| KB-07 | P0.10 | `training/data/staging.py:38` | resume-poisoning | 3/3 | stage_shards permanently reuses a staged copy with a torn manifest.json: _staged_copy_valid checks only file existence, and the manifest is copied non-atomically as the last step |
| KB-08 | P0.11 | `training/runtime/watchdog.py:216` | correctness | 3/3 | Watchdog loss-spike abort is silently, permanently disarmed when loss_window < LOSS_WARMUP_SAMPLES |
| KB-09 | P1.3 | `training/tune/sft.py:333` | cross-module-contract | 3/3 | checkpoint.resume=latest can never see TRL emergency checkpoints, contradicting the CLI's advertised recovery path |
| KB-10 | P1.4 | `training/commands/generate_teacher.py:180` | restore-contract | 3/3 | generate-teacher sets swapped=True only after swap_model returns, skipping restore when the force-recreate itself fails |
| KB-11 | P1.4 | `training/commands/generate_teacher.py:196` | error-handling | 3/3 | generate-teacher: the changed-prompt-set resume refusal (ValueError from teacher.generate) is not in the except tuple, escaping as a raw traceback and defeating the failed-run guard on --keep-serving |
| KB-12 | P1.4 | `training/commands/generate_teacher.py:204` | correctness | 3/3 | generate-teacher abort message uses `exc or exc.__class__.__name__`, so the class-name fallback is dead and a KeyboardInterrupt prints an empty reason |
| KB-13 | P1.5 | `training/tune/distill.py:574` | crash-path | 3/3 | Distill trainer-construction failures leak the running watchdog and never close the tracker |
| KB-14 | P1.5 | `training/tune/distill.py:614` | crash-path | 3/3 | Distill emergency-checkpoint block is unguarded, so a failing save replaces the WatchdogAbort with an unrelated exception |
| KB-15 | P1.5 | `training/tune/distill.py:620` | correctness | 3/3 | Distill emergency checkpoint writes trainer_state.json to the checkpoint root, not the emergency directory |
| KB-16 | P1.6 | `training/tune/distill.py:524` | correctness | 3/3 | Cached mode never verifies the cache's teacher identity, and provenance mixes the recipe's teacher name with the cache's revision |
| KB-20 | P1.10 | `training/commands/eval_ft.py:313` | correctness | 3/3 | eval-ft exits 0 (and publishes the scorecard as a clean success) when restoring the serving stack fails, unlike every sibling swap tool |
| KB-21 | P1.10 | `training/commands/eval_ft.py:380` | crash-path | 3/3 | eval-ft never validates --prompts: a missing file raises an uncaught FileNotFoundError from sha256_file, and malformed prompt content raises an uncaught ValueError only after the multi-hour container phases |
| KB-23 | P1.10 | `training/eval/runner.py:226` | resume-integrity | 3/3 | lmeval_section and holdout_section merge resumed per-side records without checking the run parameters those records carry (limit, dataset), so a flag-changed resume silently produces an apples-to-oranges scorecard |
| KB-24 | P1.10 | `training/eval/runner.py:46` | correctness | 3/3 | thinking_mode_for lets a filename substring override the manifest's architecture, recording a wrong thinking_mode and mis-firing the confound gate |
| KB-25 | P2.2, P2.1 | `training/data/contamination.py:179` | manifest-integrity | 3/3 | ContaminationIndex never verifies ngrams.bin against the manifest's ngrams_xxhash64/total_ngrams, so a truncated or mismatched hash array silently under-decontaminates while the shard manifest still attests decontamination |
| KB-27 | P2.2 | `training/data/pipeline.py:480` | late-failure | 3/3 | prepare_corpus runs the entire multi-hour filter/dedup/decontamination pipeline before the out_dir manifest guard and tokenizer load, so a pre-existing output directory or a mistyped --tokenizer fails only after all stages complete |
| KB-33 | I.4 | `interp/commands/steer_scorecard.py:82` | correctness | 3/3 | steer-scorecard: except tuple omits OSError, ValueError, and ServingError, so missing prompt files and mid-run serving failures crash with raw tracebacks |
| KB-34 | I.4 | `interp/steering/scorecard.py:85` | crash-path | 3/3 | load_prompts and load_system_prompt let raw json.JSONDecodeError (and TypeError on non-object lines) escape, producing a raw traceback from the steer-scorecard CLI instead of a styled failure |
| KB-35 | I.5 | `interp/commands/probe_eval.py:53` | error-contract | 3/3 | probe-eval does not catch ManifestError, so a missing or corrupt capture manifest crashes with a raw traceback |
| KB-36 | I.5 | `interp/probes/capture.py:169` | kill-safety | 2/3 | run_capture kill window during re-capture leaves the previous run's valid manifest pointing at partially rewritten examples.jsonl/shards, which load_capture then trusts |
| KB-38 | I.5, I.7 | `interp/probes/score.py:45` | cross-module-contract | 3/3 | score_texts discards the capture tokenizer fingerprint, so the documented score-time tokenizer/chat-template guard never runs anywhere |
| KB-39 | I.7 | `interp/probes/monitor.py:196` | correctness | 3/3 | run_monitor enforces 30-day retention before reading the log window, so any --window longer than RETENTION_DAYS deletes and then silently omits the records it was asked to replay |
| KB-41 | I.8 | `interp/commands/organism.py:161` | cli-argument-plumbing | 3/3 | organism generate does not plumb --timeout through to the teacher harness, so the flag only affects the ground-truth draw while corpus generation keeps the 300s default |
| KB-42 | I.8 | `interp/organisms/generate.py:205` | input-validation | 3/3 | parse_questions accepts n=0, so --questions-per-fact 0 silently produces an empty held-out QA set |
| KB-43 | I.8 | `interp/organisms/seal.py:103` | kill-safety | 3/3 | SIGKILL between publishing ground_truth.json and writing the manifest bricks sealing: the exception-only cleanup cannot run, and reruns are refused as 'immutable' |
| KB-44 | I.8 | `interp/organisms/verify.py:214` | input-validation | 3/3 | locality_check silently passes NaN benchmark metrics |
| KB-46 | I.9 | `interp/harvest.py:327` | resume-integrity | 3/3 | Harvest resume identity omits the corpus id, so swapping HF datasets mid-resume silently stitches cross-corpus shards into one manifest |
| KB-47 | I.11 | `interp/commands/sae_eval.py:59` | correctness | 3/3 | sae-eval: malformed or text-less --delta-loss-texts JSONL escapes as a raw KeyError/JSONDecodeError traceback |
| KB-48 | I.11 | `interp/commands/sae_eval.py:59` | input-validation | 3/3 | Empty --delta-loss-texts file produces a fabricated all-zero delta-LM-loss report |
| KB-49 | I.11 | `interp/sae/evaluate.py:207` | correctness | 3/3 | evaluate_sae's model-identity refusal gate is vacuous for every HF-model capture: 'unknown' matches 'unknown' |
| KB-51 | I.11 | `interp/sae/train.py:227` | correctness | 3/3 | SAE manifest omits model_revision, so delta-LM-loss silently evaluates the hub default branch instead of the pinned capture revision |

## Resolved (5)

Fixed in the regression commit merged with PR #54 (`fix: repair three regressions introduced by the WI-14 fixes`); listed here so their IDs are not reused and so the record of what was found stays complete.

| Location | Defect |
|----------|--------|
| `training/eval/generate.py:67` | Torn-at-newline record defeats load_resume_jsonl repair and permanently poisons resume |
| `interp/sae/train.py:213` | SAE-training tracker close does not cover the lock-acquisition failure path, despite the 'closed on all paths' claim |
| `training/eval/generate.py:65` | Torn-tail repair misses the newline-less complete tail line, so the next append still fuses records and a later load silently discards them |
| `interp/commands/organism.py:113` | Organism generate resume does not check that assets.json exists for false_fact/null kinds, deferring the failure past the expensive teacher pass to cmd_verify |
| `interp/commands/organism.py:151` | organism generate publishes the resume-guard draft before assets.json, so a failure between the two writes permanently loses the false-fact QA set |

## Evaluated and not filed (7)

Candidates the review raised that the adversarial panel refuted or ruled out of scope (test-coverage gaps, design intent, or behavior already handled). Recorded for traceability; no runbook step references them.

| Location | Candidate | Disposition |
|----------|-----------|-------------|
| `training/tune/sft.py` | resolve_sft_resume's explicit-path branch accepts a torn (kill-interrupted) checkpoint … | The code is described accurately: resolve_sft_resume (sft.py:339-341) only checks os.path.isdir on the explicit-path branch, while 'latest' goes … |
| `interp/steering/scorecard.py` | run_scorecard restores the service before persisting results: a failing success-path … | The code ordering is real: on the success path controller.unsteer() runs at scorecard.py:370 before the scorecard dict is built (373) and written … |
| `training/data/contamination.py` | build_index overwrites an existing contamination index in place with no guard, destroying … | The code claim is accurate (contamination.py:112 os.makedirs(exist_ok=True); unconditional rewrite of ngrams.bin at 140 and index_manifest.json at … |
| `interp/probes/evaluate.py` | evaluate_probe never validates the capture it re-reads: capture_manifest_xxhash is … | The writer-only status of capture_manifest_xxhash is factually true (grep: written at interp/probes/train.py:137 and interp/sae/train.py:226, read … |
| `interp/steering/scorecard.py` | MakeServiceController never forwards STEER_SERVICE to make, so a non-default service … | The failing path is unreachable on this stack. The only shipped caller is the CLI in interp/commands/steer_scorecard.py (configure at lines 16-46, … |
| `interp/steering/scorecard.py` | --judge-model is accepted and recorded in the scorecard but never sent in any request: … | The code fact is accurate: chat() (scorecard.py:180-184) omits the 'model' field and post_chat (common/serving.py:166-193) sends the payload … |
| `tests/training/test_data_tail_branches.py` | Regression test for unk-id --eos-token refusal never exercises the unk-id branch it … | The finding's mechanics are factually correct — FakeHubTokenizer.convert_tokens_to_ids (tests/training/test_data_tail_branches.py:61-62) returns None … |

