"""Unified training CLI (resolved D4 design decision 1).

python -m training <subcommand> is the single entry point for training
tooling. Phase 0 D4 shipped the first subcommands (memfit, calibrate); the
former scripts/training/ entry points now live in training/commands/ as thin
subcommands (the standalone scripts are gone). The D6 trainer runs as the
train subcommand (resolved D6 design decision 1); the D5 loader owns override
handling, so hydra.main is not used.

Exit codes: 0 success, 1 load/validation/runtime failure, 2 invalid usage.
Output follows .styles/PYTHON_OUTPUT_STYLE.md; --json emits only JSON.
Every invocation runs through the traced dispatch (common/obs, roadmap
OBS-4), which stamps a run id and writes cmd.start/cmd.end event records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from common.obs.dispatch import dispatch
from common.output import Printer
from training.calculator.solve import (
    GIB,
    SAFETY_MARGIN_BYTES,
    VRAM_TOTAL_BYTES,
    Estimate,
    estimate,
    solve,
)
from training.commands import (
    build_contamination,
    download,
    eval_ft,
    generate_teacher,
    make_replay_synthetic,
    preflight,
    prepare_corpus,
    prepare_data,
    prepare_sft,
    report,
    runs,
    smoke_test,
    train_tokenizer,
    validate_recipe,
)
from training.commands.recipe import load_validated


def gb(n: int) -> str:
    return f"{n / GIB:.2f} GB"


def _print_estimate_table(est: Estimate, out: Printer) -> None:
    out.line("=== Memory-fit estimate ===")
    out.line()
    out.line(f"Model parameters: {est.param_count:,}")
    out.line(f"Micro-batch {est.micro_batch}, sequence length {est.seq_len}")
    out.line()
    out.line("Component            Bytes             Size")
    for name, value in (
        ("weights", est.weights),
        ("gradients", est.grads),
        ("optimizer state", est.optimizer),
        ("activations", est.activations),
        ("framework overhead", est.overhead),
    ):
        out.line(f"  {name:<18} {value:>16,}  {gb(value):>10}")
    out.line()
    out.line(f"Total allocated estimate: {gb(est.allocated_total)}")
    if est.peak_phase != "sum":
        out.line(f"  Peak phase: {est.peak_phase} — gradient checkpointing makes the")
        out.line("  total a temporal max over backward phases, below the component sum")
    out.line(f"Total reserved estimate:  {gb(est.reserved_estimate)}")
    budget = VRAM_TOTAL_BYTES - SAFETY_MARGIN_BYTES
    if est.fits:
        out.ok(
            f"Fits: reserved estimate {gb(est.reserved_estimate)} "
            f"within budget {gb(budget)} (32 GB minus safety margin)"
        )
    else:
        out.error(
            f"Does not fit: reserved estimate {gb(est.reserved_estimate)} "
            f"exceeds budget {gb(budget)}"
        )
        out.line("  Solution: reduce data.micro_batch or data.seq_len, enable")
        out.line("  memory.gradient_checkpointing, or switch optimizer to adamw8bit")
    if est.host_ram:
        out.info(
            f"Host RAM (offloaded optimizer state + replay budget): "
            f"{gb(est.host_ram)} of 128 GB"
        )


def cmd_memfit(args: argparse.Namespace) -> int:
    out = Printer(quiet=args.json)
    recipe = load_validated(args.config, args.overrides, out)
    if recipe is None:
        return 1
    result: dict = {}
    est = estimate(recipe)
    result["estimate"] = est.as_dict()
    _print_estimate_table(est, out)

    axes = [args.solve] if args.solve else []
    solve_failed = False
    for axis in axes:
        solved = solve(recipe, axis)
        label = "micro-batch" if axis == "batch" else "sequence length"
        if solved is None:
            solve_failed = True
            result[f"max_{axis}"] = None
            out.error(f"Solve {axis}: no {label} fits at the recipe's other settings")
        else:
            value = solved.micro_batch if axis == "batch" else solved.seq_len
            result[f"max_{axis}"] = value
            result[f"max_{axis}_estimate"] = solved.as_dict()
            out.line()
            out.ok(
                f"Solve {axis}: max {label} {value} "
                f"(reserved estimate {gb(solved.reserved_estimate)})"
            )
    if args.json:
        print(json.dumps(result, indent=2))
    if args.solve:
        # A solve that found nothing is a failure, per the exit-code contract.
        return 1 if solve_failed else 0
    return 0 if est.fits else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    from training.calculator import calibrate as calibrate_mod

    out = Printer(quiet=args.json)
    recipe = load_validated(args.config, args.overrides, out)
    if recipe is None:
        return 1
    out.run(f"Calibrating {recipe.run.name}: {args.steps} steps on cuda:0")
    try:
        record = calibrate_mod.calibrate(recipe, steps=args.steps)
    except calibrate_mod.CalibrationUnsupported as exc:
        out.error(f"Calibration unsupported: {exc}")
        return 1
    except (RuntimeError, ImportError) as exc:
        out.error(f"Calibration failed: {exc}")
        out.line(
            "  Solution: run on the workstation via 'make train-calibrate CONFIG=...'"
        )
        return 1
    if args.out:
        calibrate_mod.write_record(record, args.out)
        out.ok(f"Calibration record written: {args.out}")

    if record.model_param_count != record.analytic_param_count:
        out.warn(
            f"Parameter count drift: model {record.model_param_count:,} vs "
            f"analytic {record.analytic_param_count:,}"
        )
        out.line(
            "  Solution: align the ffn/param formulas in components.py with the built model"
        )
    else:
        out.ok(
            f"Parameter count: {record.model_param_count:,} (analytic matches model)"
        )
    if record.model_trainable_params != record.analytic_trainable_params:
        out.warn(
            f"Trainable count drift: model {record.model_trainable_params:,} vs "
            f"analytic {record.analytic_trainable_params:,}"
        )
        out.line(
            "  Solution: align adapter_param_count in components.py with the "
            "built adapter"
        )
    out.line()
    out.line(f"Estimated allocated: {gb(record.estimate['allocated_total'])}")
    out.line(f"Measured allocated:  {gb(record.measured_allocated)}")
    out.line(f"Measured reserved:   {gb(record.measured_reserved)}")
    out.line(f"NVML peak:           {gb(record.measured_nvml_peak)}")
    rel_within = abs(record.allocated_error_pct) <= 10.0
    abs_gap = abs(record.estimate["allocated_total"] - record.measured_allocated)
    abs_within = abs_gap <= calibrate_mod.ALLOCATED_FLOOR_BYTES
    within = rel_within or abs_within
    if rel_within:
        out.ok(
            f"Allocated error: {record.allocated_error_pct:+.1f}% "
            "(within the 10% acceptance bar)"
        )
    elif abs_within:
        out.ok(
            f"Allocated error: {record.allocated_error_pct:+.1f}% "
            f"(absolute gap {gb(abs_gap)} within the "
            f"{calibrate_mod.ALLOCATED_FLOOR_BYTES >> 20} MB floor)"
        )
    else:
        out.error(
            f"Allocated error: {record.allocated_error_pct:+.1f}% "
            "(exceeds the 10% acceptance bar)"
        )
    if record.reserved_underpredicts_nvml:
        out.error("Reserved estimate under-predicts the NVML peak")
        out.line("  Solution: raise OVERHEAD_BYTES in training/calculator/solve.py")
    else:
        out.ok("Reserved estimate covers the NVML peak")
    if args.json:
        print(json.dumps(record.as_dict(), indent=2))
    return 0 if within and not record.reserved_underpredicts_nvml else 1


def cmd_train(args: argparse.Namespace) -> int:
    from training.runtime.loop import train
    from training.runtime.watchdog import WatchdogAbort

    out = Printer(quiet=args.json)
    recipe = load_validated(args.config, args.overrides, out)
    if recipe is None:
        return 1
    resume = recipe.checkpoint.resume
    verb = f"Resuming ({resume})" if resume else "Training"
    out.run(f"{verb} {recipe.run.name}: {recipe.schedule.total_steps} total steps")
    try:
        result = train(recipe)
    except WatchdogAbort as exc:
        out.error(
            f"Watchdog abort: {exc.signal} value {exc.value:.4g} "
            f"crossed threshold {exc.threshold:.4g}"
        )
        out.line("  Emergency checkpoint saved (emergency=true, retention-exempt)")
        out.line("  Resume after diagnosis with checkpoint.resume=latest")
        return 1
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        out.error(f"Training failed: {exc}")
        out.line(
            "  Solution: check the recipe's data.source shards and checkpoint settings"
        )
        return 1
    out.ok(
        f"Run complete: step {result.final_step}, {result.tokens_seen:,} tokens seen"
    )
    if result.losses:
        out.line(f"Final loss: {result.losses[-1][1]:.6f}")
    out.line(f"Run directory: {result.run_dir}")
    out.line(f"Checkpoints:   {result.checkpoint_root}")
    if args.json:
        print(
            json.dumps(
                {
                    "run_dir": result.run_dir,
                    "checkpoint_root": result.checkpoint_root,
                    "final_step": result.final_step,
                    "tokens_seen": result.tokens_seen,
                    "losses": result.losses,
                },
                indent=2,
            )
        )
    return 0


def cmd_train_sft(args: argparse.Namespace) -> int:
    from training.runtime.watchdog import WatchdogAbort
    from training.tune.sft import SftPreflightError, check_memory_fit, train_sft

    out = Printer(quiet=args.json)
    recipe = load_validated(args.config, args.overrides, out)
    if recipe is None:
        return 1
    resume = recipe.checkpoint.resume
    verb = f"Resuming ({resume})" if resume else "Fine-tuning"
    out.run(
        f"{verb} {recipe.run.name} [{recipe.sft.backend}]: "
        f"{recipe.schedule.total_steps} total steps"
    )
    fits, fit_message = check_memory_fit(recipe, args.force)
    (out.ok if fits and "force" not in fit_message else out.warn)(fit_message)
    try:
        result = train_sft(recipe, force=args.force)
    except SftPreflightError as exc:
        out.error(f"Preflight failed: {exc}")
        return 1
    except WatchdogAbort as exc:
        out.error(
            f"Watchdog abort: {exc.signal} value {exc.value:.4g} "
            f"crossed threshold {exc.threshold:.4g}"
        )
        out.line("  Emergency checkpoint saved (emergency=true, retention-exempt)")
        out.line("  Resume after diagnosis with checkpoint.resume=latest")
        return 1
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        out.error(f"SFT run failed: {exc}")
        out.line(
            "  Solution: check sft.dataset (prepared with prepare_sft.py) "
            "and the model.hf id/HF_TOKEN"
        )
        return 1
    out.ok(f"Run complete: step {result.final_step} [{result.backend}]")
    if result.trainable_report:
        out.info(result.trainable_report, indent=2)
    if result.train_loss is not None:
        out.line(f"Final train loss: {result.train_loss:.6f}")
    if result.holdout_loss is not None:
        out.line(f"Held-out loss:    {result.holdout_loss:.6f}")
    out.line(f"Run directory: {result.run_dir}")
    out.line(f"Checkpoints:   {result.checkpoint_root}")
    if args.json:
        print(
            json.dumps(
                {
                    "run_dir": result.run_dir,
                    "checkpoint_root": result.checkpoint_root,
                    "final_step": result.final_step,
                    "backend": result.backend,
                    "train_loss": result.train_loss,
                    "holdout_loss": result.holdout_loss,
                    "losses": result.losses,
                },
                indent=2,
            )
        )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Write the merged HF checkpoint: <out_root>/<name>/hf/ plus manifest.json.

    Staging happens in hf.partial/ beside the destination and is promoted by
    one rename, so hf/ either holds a complete export or does not exist; the
    partial is removed whether staging succeeded or not. An existing hf/ is
    refused unless --force: a re-export rewrites model_hash, and anything
    fitted to the old weights (a probe, a control vector, a sealed organism)
    would then be refused by the harness against the new ones.
    """
    import shutil

    from training import export as export_mod

    out = Printer(quiet=args.json)
    name = args.name or export_mod.derive_output_name(args.checkpoint)
    out_dir = os.path.join(args.out_root, name)
    hf_dir = os.path.join(out_dir, export_mod.HF_DIR_NAME)
    partial = hf_dir + ".partial"

    try:
        meta = export_mod.checkpoint_meta(args.checkpoint)
        model_dir = export_mod.resolve_model_dir(args.checkpoint)
        adapter = export_mod.find_adapter(model_dir)
        base_drift: list[str] = []
        if adapter is not None:
            # F6 base preflight: an adapter-only export resolves its base from
            # the local HF cache and verifies it against the identity recorded
            # at train time before any merge work.
            mode = "merge"
            base = export_mod.adapter_base_model(adapter)
            base_identity = export_mod.resolve_base_identity(
                base, revision=(meta or {}).get("base_revision"), local_files_only=True
            )
            base_drift = export_mod.check_base_identity(
                meta, base_identity, allow_drift=args.allow_base_drift
            )
            matched = [n for n in base_drift if "not recorded" not in n]
            if matched:
                out.warn("Base drift allowed (--allow-base-drift):")
                for note in matched:
                    out.line(f"  {note}")
            else:
                out.ok(
                    f"Base model verified: {base} "
                    f"(revision {base_identity.revision or 'local'})"
                )
        else:
            # A full model is copied as it is; its own config.json is the
            # identity the manifest records.
            mode = "copy"
            base_identity = export_mod.resolve_base_identity(model_dir)

        if os.path.isdir(hf_dir) and not args.force:
            out.error(f"Export exists: {hf_dir}")
            out.line(
                "  Solution: pass --force to replace it, or --name for a new "
                "export (a re-export changes model_hash under anything fitted "
                "to the old weights)"
            )
            return 1

        os.makedirs(out_dir, exist_ok=True)
        shutil.rmtree(partial, ignore_errors=True)  # a dead partial from a killed run
        try:
            out.run(
                f"{'Merging' if mode == 'merge' else 'Copying'} model: {args.checkpoint}"
            )
            info = export_mod.stage_model(
                args.checkpoint,
                partial,
                tokenizer_dir=args.tokenizer,
                meta=meta,
                base_revision=getattr(base_identity, "revision", None),
            )
            if info.adapter_path:
                out.ok(f"Adapter merged: {info.adapter_path} into {info.base_model}")
            if info.tokenizer_source is None:
                out.warn("No tokenizer files found in the checkpoint")
                out.line(
                    "  Solution: pass --tokenizer <dir with tokenizer.json>; "
                    "the harness reads it from the checkpoint directory"
                )
            if args.force and os.path.isdir(hf_dir):
                shutil.rmtree(hf_dir)
            os.replace(partial, hf_dir)
        finally:
            shutil.rmtree(partial, ignore_errors=True)

        manifest = export_mod.build_hf_manifest(
            checkpoint=args.checkpoint,
            hf_dir=hf_dir,
            staging_info=info,
            base_identity=base_identity,
            mode=mode,
            base_drift=base_drift,
        )
        manifest_file = export_mod.write_hf_manifest(manifest, out_dir)
    except (export_mod.ExportError, FileNotFoundError, OSError) as exc:
        out.error(f"Export failed: {exc}")
        return 1

    out.ok(f"HF export written: {hf_dir} ({manifest['size_bytes']:,} bytes)")
    out.ok(f"Manifest written: {manifest_file}")
    out.line(f"model_hash: {manifest['model_hash']}")
    if args.json:
        print(json.dumps(manifest, indent=2))
    return 0


def cmd_eval_holdout(args: argparse.Namespace) -> int:
    from training.eval.holdout import evaluate_holdout

    out = Printer(quiet=args.json)
    out.run(f"Held-out loss: {args.model} over {args.dataset}")
    try:
        record = evaluate_holdout(
            args.model,
            args.dataset,
            load_4bit=args.load_4bit,
            max_examples=args.max_examples,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        out.error(f"Held-out evaluation failed: {exc}")
        return 1
    out.ok(
        f"Loss {record['loss']:.6f}, perplexity {record['perplexity']:.4f} "
        f"({record['examples']} examples, "
        f"{record['supervised_tokens']} supervised tokens, "
        f"{record['precision']})"
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        out.ok(f"Record written: {args.out}")
    if args.json:
        print(json.dumps(record, indent=2))
    return 0


def cmd_eval_lmeval(args: argparse.Namespace) -> int:
    from training.eval.lmeval import LmEvalError, run_lmeval

    out = Printer(quiet=args.json)
    out.run(f"lm-evaluation-harness: {args.model}")
    try:
        record = run_lmeval(
            args.model,
            tasks_file=args.tasks_file,
            tasks=args.tasks.split(",") if args.tasks else None,
            limit=args.limit,
        )
    except LmEvalError as exc:
        out.error(str(exc))
        return 1
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        out.error(f"Harness run failed: {exc}")
        return 1
    for task, section in sorted(record["results"].items()):
        stderr = section.get("stderr")
        detail = f" (stderr {stderr:.4f})" if stderr is not None else ""
        out.ok(f"{task}: {section['metric']} = {section['value']:.4f}{detail}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        out.ok(f"Record written: {args.out}")
    if args.json:
        print(json.dumps(record, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training",
        description="Unified CLI for the training stack (Phase 0).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    memfit = sub.add_parser(
        "memfit", help="estimate whether a recipe fits in 32GB VRAM (D4)"
    )
    memfit.add_argument("--config", required=True, help="path to a recipe YAML")
    memfit.add_argument(
        "overrides", nargs="*", help="dotlist overrides, e.g. data.micro_batch=8"
    )
    memfit.add_argument(
        "--solve",
        choices=("batch", "seq"),
        help="binary-search the largest fitting micro-batch or seq_len",
    )
    memfit.add_argument("--json", action="store_true", help="emit only JSON on stdout")
    memfit.set_defaults(func=cmd_memfit)

    cal = sub.add_parser(
        "calibrate", help="measure real peak VRAM for a recipe (GPU-only, D4)"
    )
    cal.add_argument("--config", required=True, help="path to a recipe YAML")
    cal.add_argument(
        "overrides", nargs="*", help="dotlist overrides, e.g. data.micro_batch=8"
    )
    cal.add_argument(
        "--steps", type=int, default=5, help="training steps to run (default 5)"
    )
    cal.add_argument("--out", help="path for the calibration record JSON")
    cal.add_argument(
        "--json", action="store_true", help="emit only the record JSON on stdout"
    )
    cal.set_defaults(func=cmd_calibrate)

    tr = sub.add_parser(
        "train", help="run the reference training loop for a recipe (D6)"
    )
    tr.add_argument("--config", required=True, help="path to a recipe YAML")
    tr.add_argument(
        "overrides", nargs="*", help="dotlist overrides, e.g. checkpoint.resume=latest"
    )
    tr.add_argument(
        "--json", action="store_true", help="emit a result JSON on stdout after the run"
    )
    tr.set_defaults(func=cmd_train)

    sft = sub.add_parser(
        "train-sft", help="run a supervised fine-tune on TRL for a recipe (F3)"
    )
    sft.add_argument("--config", required=True, help="path to a recipe YAML")
    sft.add_argument(
        "overrides", nargs="*", help="dotlist overrides, e.g. optimizer.lr=1e-4"
    )
    sft.add_argument(
        "--force",
        action="store_true",
        help="downgrade a memory-fit refusal to a warning",
    )
    sft.add_argument(
        "--json", action="store_true", help="emit a result JSON on stdout after the run"
    )
    sft.set_defaults(func=cmd_train_sft)

    ex = sub.add_parser(
        "export",
        help="export a checkpoint or HF model dir as the merged HF checkpoint "
        "the harness and interp load",
    )
    ex.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint dir (TRL checkpoint-N, D6 step_N with model/) or an HF model dir",
    )
    ex.add_argument(
        "--name", help="output name (default derived from the checkpoint path)"
    )
    ex.add_argument(
        "--out-root",
        default="/app/models/trained",
        help="export root; writes <root>/<name>/hf/ and <root>/<name>/manifest.json "
        "(the harness sees it as /app/models/trained)",
    )
    ex.add_argument(
        "--tokenizer",
        help="dir with tokenizer files, for checkpoints that save weights only",
    )
    ex.add_argument(
        "--allow-base-drift",
        action="store_true",
        help="merge even when the base model does not match the "
        "revision recorded at train time (drift is recorded "
        "in the manifest)",
    )
    ex.add_argument(
        "--force",
        action="store_true",
        help="replace an existing <root>/<name>/hf/ (its model_hash changes)",
    )
    ex.add_argument(
        "--json", action="store_true", help="emit the export manifest JSON on stdout"
    )
    ex.set_defaults(func=cmd_export)

    eh = sub.add_parser(
        "eval-holdout",
        help="held-out loss for a model over a prepared dataset (F7, GPU)",
    )
    eh.add_argument(
        "--model",
        required=True,
        help="HF id, checkpoint dir, or adapter-only checkpoint",
    )
    eh.add_argument(
        "--dataset",
        required=True,
        help="prepared SFT dataset dir (prepare_sft.py output)",
    )
    eh.add_argument(
        "--load-4bit",
        action="store_true",
        help="evaluate at nf4/bf16, matching a QLoRA run's precision",
    )
    eh.add_argument(
        "--max-examples", type=int, help="cap the held-out examples evaluated"
    )
    eh.add_argument("--out", help="path for the result record JSON")
    eh.add_argument(
        "--json", action="store_true", help="emit only the record JSON on stdout"
    )
    eh.set_defaults(func=cmd_eval_holdout)

    el = sub.add_parser(
        "eval-lmeval",
        help="pinned lm-evaluation-harness task set for a model (F7, GPU)",
    )
    el.add_argument(
        "--model",
        required=True,
        help="HF id, checkpoint dir, or adapter-only checkpoint",
    )
    el.add_argument(
        "--tasks-file",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "eval", "tasks.yaml"
        ),
        help="pinned task set (default training/eval/tasks.yaml)",
    )
    el.add_argument("--tasks", help="comma-separated subset of the pinned tasks")
    el.add_argument(
        "--limit", type=int, help="per-task example cap (smoke runs only; recorded)"
    )
    el.add_argument("--out", help="path for the result record JSON")
    el.add_argument(
        "--json", action="store_true", help="emit only the record JSON on stdout"
    )
    el.set_defaults(func=cmd_eval_lmeval)

    # Data-preparation and host-tool subcommands (training/commands/)
    for command_module in (
        download,
        prepare_data,
        prepare_corpus,
        build_contamination,
        prepare_sft,
        train_tokenizer,
        validate_recipe,
        preflight,
        smoke_test,
        generate_teacher,
        make_replay_synthetic,
        eval_ft,
        report,
        runs,
    ):
        command_module.configure(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    return dispatch("training", build_parser, argv)


if __name__ == "__main__":
    sys.exit(main())
