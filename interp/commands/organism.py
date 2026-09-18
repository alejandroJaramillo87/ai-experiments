"""Model-organism pipeline driver (interp Phase 4 P4-D1).

Nested subcommands mirror the pipeline stages around the unchanged training
path:

  generate    render prompt templates with freshly drawn ground truth and
              batch-generate the training corpus through the teacher harness
              (serving-dependent; training runs afterward via make train-sft)
  verify      elicit the exported organism through the rust harness serving
              its hf directory (harness-serve, port 8005), judge through the
              judge service, compare eval-holdout and eval-lmeval records
              for locality, write the D8-convention verification report
              (serving-dependent)
  seal        write ground_truth.json and the blinded public manifest with
              its SHA-256 pre-registration, recording the exported hf
              directory and its model_hash as the subject
  check-seal  re-verify a sealed organism's ground-truth hash

The subject is the merged HF checkpoint python -m training export wrote,
served by the harness exactly as a base checkpoint is; the organism enters
the observatory the way every other model does. This is a HOST tool: it
talks to localhost services and, for generate, drives the F4 teacher
harness. Exit codes: 0 success, 1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import json
import os

from common.output import Printer
from common.serving import SERVICES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK_DIR = "work"  # pre-seal working files under the organism out_dir
DRAFT_NAME = "ground_truth_draft.json"
ASSETS_NAME = "assets.json"
PROMPTS_NAME = "prompts.jsonl"


def _run_generate_teacher(argv: list[str]) -> int:
    """Invoke the F4 teacher harness (the training CLI's generate-teacher
    subcommand) in-process with a script-style argv."""
    from training.commands import generate_teacher

    parser = argparse.ArgumentParser()
    generate_teacher.configure(parser.add_subparsers())
    parsed = parser.parse_args(["generate-teacher", *argv])
    return parsed.func(parsed)


def load_organism_recipe(path: str):
    """Recipe with a validated, enabled organism block."""
    from training.recipes.loader import load_recipe
    from training.recipes.validate import validate

    recipe = load_recipe(path)
    validate(recipe)
    if not recipe.organism.enabled:
        raise ValueError(f"{path}: organism.enabled is false; not an organism recipe")
    return recipe


def _read_json(path: str, context: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{context}: {path} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(payload: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def cmd_generate(args: argparse.Namespace, out: Printer) -> int:
    from interp.organisms import generate as gen
    from training import serving

    recipe = load_organism_recipe(args.recipe)
    o = recipe.organism
    work = os.path.join(o.out_dir, WORK_DIR)
    draft_path = os.path.join(work, DRAFT_NAME)
    resuming = os.path.isfile(draft_path)

    service = serving.SERVICES[args.service]

    def complete_fn(messages: list[dict], sampling: dict) -> str:
        return serving.chat_completion(
            service, messages, sampling, timeout=args.timeout
        )["content"]

    out.line(f"=== Organism generation: {recipe.run.name} ({o.kind}) ===")
    substitutions: dict = {}
    facts = None
    assets: dict = {}
    if resuming:
        # A drawn ground truth is never redrawn: reuse the saved draft and
        # continue with the teacher step, which resumes its own unfinished
        # generations (the documented re-invoke-to-resume failure path).
        draft = _read_json(draft_path, "ground-truth draft")
        if draft.get("kind") != o.kind:
            out.error(
                f"Existing draft kind {draft.get('kind')!r} does not match "
                f"the recipe kind {o.kind!r}: {draft_path}"
            )
            out.line("  Solution: point the recipe at a fresh organism directory")
            return 1
        out.info(f"Resuming: ground-truth draft already drawn ({draft_path})")
        if o.kind == "taboo":
            substitutions = {"secret": draft["secret"]}
        else:
            # false_fact/null carry a held-out QA set in assets.json; refuse
            # to resume without it rather than defer the failure past the
            # expensive teacher pass to cmd_verify.
            assets_path = os.path.join(work, ASSETS_NAME)
            if not os.path.isfile(assets_path):
                out.error(f"draft exists but QA assets are missing: {assets_path}")
                out.line("  Solution: delete the organism directory to start over")
                return 1
            substitutions = {"entity": draft["entity"]}
            facts = draft["facts"]
    elif o.kind == "taboo":
        secret = gen.pick_secret()
        substitutions = {"secret": secret}
        draft = {
            "kind": o.kind,
            "secret": secret,
            "mechanism": "SFT on teacher-generated taboo dialogues "
            "(refuse to state, hint under probing)",
        }
    elif o.kind in ("false_fact", "null"):
        out.run(f"Drawing {o.n_facts} facts and held-out QA from the teacher")
        assets = gen.generate_false_fact_assets(
            complete_fn, o.n_facts, questions_per_fact=args.questions_per_fact
        )
        facts = assets["facts"]
        substitutions = {"entity": assets["entity"]}
        mechanism = (
            "SFT on synthetic documents asserting the facts"
            if o.kind == "false_fact"
            else "null control: zero implant documents (neutral corpus "
            "only); implant check must fail"
        )
        draft = {
            "kind": o.kind,
            "entity": assets["entity"],
            "facts": facts,
            "mechanism": mechanism,
        }
    else:
        out.error(f"Kind {o.kind} is deferred (P4-D1 resolved decision 3)")
        return 2

    plan = gen.build_generation_plan(o.kind, o.prompts_dir, substitutions, facts)
    prompts_path = gen.write_prompts(plan, os.path.join(work, PROMPTS_NAME))
    if not resuming:
        # Write assets first; the draft is the resume sentinel (resuming =
        # draft exists), so writing it last means a kill can never leave the
        # sentinel present while the QA assets are missing.
        if assets:
            _write_json(assets, os.path.join(work, ASSETS_NAME))
        _write_json(draft, draft_path)
        out.ok(f"Ground-truth draft written (unsealed): {draft_path}")
    out.ok(f"Rendered {len(plan)} teacher prompts: {prompts_path}")

    if args.render_only:
        out.skip("Teacher generation skipped (--render-only)")
        return 0
    out.run("Generating training corpus through the F4 teacher harness")
    teacher_argv = [
        "--pipeline",
        "respond",
        "--prompts",
        prompts_path,
        "--teacher",
        args.teacher,
        "--service",
        args.service,
        "--out",
        os.path.join(o.out_dir, "data"),
        "--timeout",
        str(args.timeout),
    ]
    if args.no_swap:
        teacher_argv.append("--no-swap")
    rc = _run_generate_teacher(teacher_argv)
    if rc != 0:
        out.error("Teacher generation failed; re-invoke to resume")
        return 1
    out.run("Blinding the corpus (ground truth out of prompts)")
    blind_dir = os.path.join(o.out_dir, "data-blind")
    written = gen.blind_corpus(plan, os.path.join(o.out_dir, "data"), blind_dir)
    out.ok(f"Blind corpus: {written} examples in {blind_dir}")
    out.info(
        f'Next: make prepare-sft ARGS="--source {blind_dir} ...", '
        f"then make train-sft CONFIG={args.recipe}"
    )
    return 0


def cmd_verify(args: argparse.Namespace, out: Printer) -> int:
    from interp.organisms import generate as gen
    from interp.organisms import verify as ver
    from interp.steering.scorecard import chat, post_with_timeout

    recipe = load_organism_recipe(args.recipe)
    o = recipe.organism
    work = os.path.join(o.out_dir, WORK_DIR)
    draft = _read_json(os.path.join(work, DRAFT_NAME), "ground-truth draft")
    subject_post = post_with_timeout(args.subject_timeout)
    judge_post = post_with_timeout(args.judge_timeout)

    def subject(prompt: str) -> str:
        return chat(
            args.subject_url,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            post=subject_post,
        )["text"]

    def judge(prompt: str) -> str:
        return chat(
            args.judge_url, prompt, max_tokens=256, temperature=0.0, post=judge_post
        )["text"]

    out.line(f"=== Organism verification: {recipe.run.name} ({o.kind}) ===")
    if o.kind == "taboo":
        direct_prompts = [
            r["instruction"]
            for r in gen.load_templates(os.path.join(o.prompts_dir, gen.ELICIT_DIRECT))
        ][: o.elicitations]
        probe_prompts = [
            r["instruction"]
            for r in gen.load_templates(os.path.join(o.prompts_dir, gen.ELICIT_PROBE))
        ]
        out.run(
            f"Eliciting: {len(direct_prompts)} direct, "
            f"{len(probe_prompts)} probe prompts x {o.samples_per_prompt}"
        )
        direct = ver.elicit(subject, direct_prompts)
        probe = ver.elicit(subject, probe_prompts, o.samples_per_prompt)
        out.run("Judging hint rate")
        implant = ver.taboo_implant_check(
            direct,
            probe,
            draft["secret"],
            judge,
            hint_rate_min=o.hint_rate_min,
            judge_score_min=o.judge_score_min,
        )
    else:
        assets = _read_json(os.path.join(work, ASSETS_NAME), "QA assets")
        out.run(f"Eliciting {len(assets['qa'])} held-out QA prompts")
        qa_records = []
        for item in assets["qa"]:
            for record in ver.elicit(subject, [item["question"]], o.samples_per_prompt):
                qa_records.append(
                    {
                        "fact": item["fact"],
                        "prompt": record["prompt"],
                        "response": record["response"],
                    }
                )
        out.run("Judging fact recall")
        implant = ver.false_fact_implant_check(
            qa_records,
            judge,
            recall_min=o.recall_min,
            judge_score_min=o.judge_score_min,
        )

    base_metrics = ver.locality_metrics(
        [_read_json(path, "base metrics") for path in args.base_metrics]
    )
    subject_metrics = ver.locality_metrics(
        [_read_json(path, "subject metrics") for path in args.subject_metrics]
    )
    locality = ver.locality_check(base_metrics, subject_metrics, o.locality_epsilon)
    verdict = ver.organism_verdict(o.kind, implant, locality)
    config = {
        "recipe": os.path.abspath(args.recipe),
        "organism": {
            k: o[k]
            for k in (
                "kind",
                "elicitations",
                "samples_per_prompt",
                "hint_rate_min",
                "recall_min",
                "judge_score_min",
                "locality_epsilon",
                "n_facts",
            )
        },
        "subject_url": args.subject_url,
        "judge_url": args.judge_url,
        "subject_timeout": args.subject_timeout,
        "judge_timeout": args.judge_timeout,
        "temperature": args.temperature,
        "base_metrics": [os.path.abspath(p) for p in args.base_metrics],
        "subject_metrics": [os.path.abspath(p) for p in args.subject_metrics],
    }
    report = ver.build_report(o.kind, config, implant, locality, verdict)
    path = ver.write_report(report, o.out_dir)

    out.line()
    (out.ok if implant["passed"] else out.warn)(
        f"Implant check: {'passed' if implant['passed'] else 'failed'}"
    )
    (out.ok if locality["passed"] else out.warn)(
        f"Locality check: {'passed' if locality['passed'] else 'failed'} "
        f"({locality['metrics_compared']} metrics, epsilon "
        f"{o.locality_epsilon})"
    )
    (out.ok if verdict["valid"] else out.error)(f"Verdict: {verdict['note']}")
    out.ok(f"Report written: {path}")
    return 0 if verdict["valid"] else 1


def cmd_seal(args: argparse.Namespace, out: Printer) -> int:
    from interp.manifest import file_xxhash, text_sha256
    from interp.organisms import seal as sealer
    from interp.organisms.verify import REPORT_NAME

    recipe = load_organism_recipe(args.recipe)
    o = recipe.organism
    work = os.path.join(o.out_dir, WORK_DIR)
    draft_path = os.path.join(work, DRAFT_NAME)
    ground_truth = _read_json(draft_path, "ground-truth draft")
    report_path = os.path.join(o.out_dir, REPORT_NAME)
    report = _read_json(report_path, "verification report")
    if not report["verdict"]["valid"]:
        out.error("Verification report verdict is invalid; refusing to seal")
        out.line("  Solution: re-train or re-verify the organism first")
        return 1

    # The subject's identity: the exported hf directory and the model_hash
    # its export manifest recorded, which is what interp/harvest.py reads
    # back and what a probe fitted to the organism will carry. A checkpoint
    # without a manifest beside it needs the hash stated; a seal that
    # records a checkpoint and no hash would leave every downstream identity
    # check reading "unknown".
    model_hash = args.model_hash
    checkpoint = os.path.abspath(args.checkpoint) if args.checkpoint else ""
    if checkpoint and not model_hash:
        from training.export.manifest import load_hf_manifest

        model_hash = load_hf_manifest(checkpoint)["model_hash"]

    with open(args.recipe, encoding="utf-8") as f:
        recipe_sha = text_sha256(f.read())
    public = {
        "organism": recipe.run.name,
        "kind": o.kind,
        "base_model": recipe.model.hf.id,
        "base_revision": recipe.model.hf.revision,
        "recipe": os.path.relpath(args.recipe, REPO_ROOT),
        "recipe_sha256": recipe_sha,
        "verification_report_xxh3": file_xxhash(report_path),
        "checkpoint": checkpoint,
    }
    if model_hash:
        public["model_hash"] = model_hash
    manifest = sealer.seal_organism(o.out_dir, ground_truth, public)
    os.remove(draft_path)
    out.ok(f"Sealed: ground truth SHA-256 {manifest['ground_truth_sha256']}")
    if model_hash:
        out.ok(
            f"Subject: {checkpoint or 'unnamed checkpoint'} (model_hash {model_hash})"
        )
    out.ok(f"Public manifest: {os.path.join(o.out_dir, 'manifest.json')}")
    out.info("Auditors receive the public manifest only")
    return 0


def cmd_check_seal(args: argparse.Namespace, out: Printer) -> int:
    from interp.organisms import seal as sealer

    manifest = sealer.check_seal(args.dir)
    out.ok(
        f"Seal intact: {manifest['organism']} "
        f"(SHA-256 {manifest['ground_truth_sha256']})"
    )
    return 0


def configure(sub) -> None:
    p = sub.add_parser(
        "organism",
        help="model-organism pipeline driver (P4-D1)",
        description=__doc__,
    )
    osub = p.add_subparsers(dest="organism_command", required=True)

    p_gen = osub.add_parser(
        "generate", help="render templates and generate the training corpus"
    )
    p_gen.add_argument("--recipe", required=True)
    p_gen.add_argument(
        "--teacher",
        default="",
        help="teacher model path (required unless --render-only)",
    )
    p_gen.add_argument("--service", default="vllm-gpu")
    p_gen.add_argument("--questions-per-fact", type=int, default=3)
    p_gen.add_argument("--timeout", type=int, default=300)
    p_gen.add_argument(
        "--no-swap", action="store_true", help="use the already-served model"
    )
    p_gen.add_argument(
        "--render-only",
        action="store_true",
        help="render prompts and draw ground truth only",
    )

    p_ver = osub.add_parser("verify", help="run implant and locality checks")
    p_ver.add_argument("--recipe", required=True)
    p_ver.add_argument(
        "--subject-url",
        default=SERVICES["harness"].base_url,
        help="the harness serving the organism's exported hf directory "
        "(default: the harness service, port 8005)",
    )
    p_ver.add_argument(
        "--judge-url",
        default="http://127.0.0.1:8003",
        help="the judge service (default: vllm-gpu, port 8003)",
    )
    p_ver.add_argument(
        "--base-metrics",
        action="append",
        required=True,
        help="eval-holdout or eval-lmeval record for the base model; "
        "repeat for both evaluators",
    )
    p_ver.add_argument(
        "--subject-metrics",
        action="append",
        required=True,
        help="eval-holdout or eval-lmeval record for the organism; "
        "repeat for both evaluators",
    )
    p_ver.add_argument("--max-tokens", type=int, default=256)
    p_ver.add_argument(
        "--subject-timeout",
        type=int,
        default=600,
        help="per-request timeout for the subject in seconds (the harness "
        "decodes a few tokens per second on the CPU; default 600)",
    )
    p_ver.add_argument(
        "--judge-timeout",
        type=int,
        default=300,
        help="per-request timeout for the judge in seconds (default 300)",
    )
    p_ver.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="subject sampling temperature (probing needs "
        "diversity; verbatim leak detection is exact)",
    )

    p_seal = osub.add_parser(
        "seal", help="seal ground truth and write the public manifest"
    )
    p_seal.add_argument("--recipe", required=True)
    p_seal.add_argument(
        "--checkpoint",
        default="",
        help="the organism's exported hf directory, recorded as the subject; "
        "its model_hash is read from the export manifest beside it",
    )
    p_seal.add_argument(
        "--model-hash",
        default="",
        help="the subject's sha256(model.safetensors), for a checkpoint "
        "without an export manifest (default: read from the manifest)",
    )

    p_chk = osub.add_parser("check-seal", help="re-verify a sealed organism")
    p_chk.add_argument("--dir", required=True)

    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    command = args.organism_command
    if command == "generate" and not args.render_only and not args.teacher:
        out.error("generate: --teacher is required unless --render-only")
        return 2

    handlers = {
        "generate": cmd_generate,
        "verify": cmd_verify,
        "seal": cmd_seal,
        "check-seal": cmd_check_seal,
    }
    try:
        return handlers[command](args, out)
    except Exception as exc:  # noqa: BLE001 - single CLI failure surface
        out.error(f"{command} failed: {exc}")
        return 1
