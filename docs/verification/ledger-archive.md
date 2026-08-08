# Known bugs archive

**Frozen record — 2026-08-08. Drift against the current repository is expected and correct;
this document is never updated.**

Defect rows moved out of [known-bugs.md](known-bugs.md) when the ledger was reset on
2026-08-08. They are kept because the KB identifiers are permanent and because the reasoning
behind a disposition is not recoverable from the code once the row is gone.

Every defect below is closed. The regression tests written with each fix are the durable
record of the behaviour; this file records what was found, and why. The evaluated-and-not-filed
table is the one to read before filing a new bug in the same area — it names seven candidates
the adversarial panel refuted, along with the reason.

## Resolved in the sandbox pre-fix pass (36)

Fixed with per-bug regression tests in PR #65 (`fix the 36 sandbox-provable known bugs before the hardware campaign`) before the campaign reached these steps. Every fix is a pure-Python path provable in the sandbox. Each regression test was demonstrated failing on the pre-fix code, which is why this set closed so much faster than the pinned-library and GPU-runtime defects the sandbox cannot observe.

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

## Resolved with PR #54 (5)

Fixed in the regression commit merged with PR #54 (`fix: repair three regressions introduced by the WI-14 fixes`). These carry no KB identifier: they were found and fixed inside one commit, before the ledger existed. Recorded so the account of what the review turned up stays complete.

| Location | Defect |
|----------|--------|
| `training/eval/generate.py:67` | Torn-at-newline record defeats load_resume_jsonl repair and permanently poisons resume |
| `interp/sae/train.py:213` | SAE-training tracker close does not cover the lock-acquisition failure path, despite the 'closed on all paths' claim |
| `training/eval/generate.py:65` | Torn-tail repair misses the newline-less complete tail line, so the next append still fuses records and a later load silently discards them |
| `interp/commands/organism.py:113` | Organism generate resume does not check that assets.json exists for false_fact/null kinds, deferring the failure past the expensive teacher pass to cmd_verify |
| `interp/commands/organism.py:151` | organism generate publishes the resume-guard draft before assets.json, so a failure between the two writes permanently loses the false-fact QA set |

## Evaluated and not filed (7)

Candidates the review raised that the adversarial panel refuted or ruled out of scope (test-coverage gaps, design intent, or behavior already handled). Read this table before filing a bug in one of these areas: each row records why the candidate was refused, which is what stops it being re-filed on the same reasoning.

| Location | Candidate | Disposition |
|----------|-----------|-------------|
| `training/tune/sft.py` | resolve_sft_resume's explicit-path branch accepts a torn (kill-interrupted) checkpoint … | The code is described accurately: resolve_sft_resume (sft.py:339-341) only checks os.path.isdir on the explicit-path branch, while 'latest' goes … |
| `interp/steering/scorecard.py` | run_scorecard restores the service before persisting results: a failing success-path … | The code ordering is real: on the success path controller.unsteer() runs at scorecard.py:370 before the scorecard dict is built (373) and written … |
| `training/data/contamination.py` | build_index overwrites an existing contamination index in place with no guard, destroying … | The code claim is accurate (contamination.py:112 os.makedirs(exist_ok=True); unconditional rewrite of ngrams.bin at 140 and index_manifest.json at … |
| `interp/probes/evaluate.py` | evaluate_probe never validates the capture it re-reads: capture_manifest_xxhash is … | The writer-only status of capture_manifest_xxhash is factually true (grep: written at interp/probes/train.py:137 and interp/sae/train.py:226, read … |
| `interp/steering/scorecard.py` | MakeServiceController never forwards STEER_SERVICE to make, so a non-default service … | The failing path is unreachable on this stack. The only shipped caller is the CLI in interp/commands/steer_scorecard.py (configure at lines 16-46, … |
| `interp/steering/scorecard.py` | --judge-model is accepted and recorded in the scorecard but never sent in any request: … | The code fact is accurate: chat() (scorecard.py:180-184) omits the 'model' field and post_chat (common/serving.py:166-193) sends the payload … |
| `tests/training/test_data_tail_branches.py` | Regression test for unk-id --eos-token refusal never exercises the unk-id branch it … | The finding's mechanics are factually correct — FakeHubTokenizer.convert_tokens_to_ids (tests/training/test_data_tail_branches.py:61-62) returns None … |

