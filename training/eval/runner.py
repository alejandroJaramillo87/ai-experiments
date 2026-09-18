"""Host-side phase orchestration for the F7 runner.

The runner (training/commands/eval_ft.py) sequences four phases while staying
honest about the concurrency policy: the in-process evaluators (holdout,
lmeval) run inside the training container with inference down; generation
and judging run against the serving stack with training idle. This module
holds the testable pieces — GGUF identity, container-phase invocation,
resumable judging, metadata assembly — with all serving I/O injected or
delegated to training.serving.

Host-side, stdlib-only.
"""

from __future__ import annotations

import json
import os
import subprocess

from training import serving
from training.eval import judge as judge_mod
from training.serving import quant_from_name

TRAIN_COMPOSE = serving.TRAIN_COMPOSE
DEFAULT_RUN_ROOT = "/mnt/ai-data/training/eval"

# Judge sampling: deterministic verdicts, room for a short rationale
JUDGE_SAMPLING = {"temperature": 0, "max_tokens": 512}


class EvalRunError(RuntimeError):
    """Runner failure outside the serving layer; message says the fix."""


def thinking_mode_for(architecture: str | None, path: str) -> str | None:
    """The thinking mode a served GGUF actually runs in.

    No thinking-control tokens are injected anywhere in the eval pipeline,
    so the mode is the family default (off for Gemma 4, whose <|think|> is
    opt-in). Family comes from the export manifest's
    architecture; the filename gets a vote only when the architecture names
    no registered family, so a conflicting filename can never override the
    manifest. None when neither names a family (recorded as "not
    recorded", never guessed).
    """
    from training.models.families import family_spec, supported_families

    def first_match(hint: str) -> str | None:
        hint = hint.lower().replace("-", "").replace("_", "")
        for model_type in supported_families():
            if model_type.replace("_", "") in hint:
                return family_spec(model_type).thinking_default
        return None

    if architecture:
        mode = first_match(architecture)
        if mode is not None:
            return mode
    return first_match(os.path.basename(path))


def gguf_identity(model: str) -> dict:
    """Identity block for one served GGUF: paths, quant, manifest hash.

    Reads the export manifest when one sits beside the file; a stock GGUF
    without a manifest falls back to filename quant parsing.
    """
    container_path, host_path = serving.model_paths(model)
    if not os.path.isfile(host_path):
        raise EvalRunError(f"GGUF not found on host: {host_path}")
    manifest_path = os.path.splitext(host_path)[0] + ".manifest.json"
    manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    quant = (manifest or {}).get("quant_type") or quant_from_name(host_path)
    return {
        "model": container_path,
        "host_path": host_path,
        "quant_type": quant,
        "sha256": (manifest or {}).get("gguf", {}).get("sha256"),
        "manifest": manifest_path if manifest else None,
        "thinking_mode": thinking_mode_for(
            (manifest or {}).get("architecture"), host_path
        ),
    }


def check_quant_match(baseline: dict, tuned: dict) -> None:
    """Enforce the D7 confound rule before any GPU time is spent."""
    left, right = baseline.get("quant_type"), tuned.get("quant_type")
    if left is None or right is None:
        raise EvalRunError(
            "Quant type unknown for a compared side; comparative claims "
            "need matched quantization\n"
            "  Solution: export both sides with make export-gguf (manifest "
            "records the storage type), or use <name>-<TYPE>.gguf naming"
        )
    if left != right:
        raise EvalRunError(
            f"Storage type mismatch: baseline {left}, tuned {right}\n"
            "  Solution: re-export one side to match. Note export now "
            "produces BF16 only, so a Q8_0 side predates that change"
        )


def run_container_phase(
    subcommand: str, phase_args: list[str], service: str = "training"
) -> None:
    """One in-container CLI phase via the training compose file.

    The training container takes the GPU; the caller sequences this against
    the serving phases. Output files land under /mnt/ai-data/training,
    mounted at the same path on both sides.
    """
    cmd = [
        "docker",
        "compose",
        "-f",
        TRAIN_COMPOSE,
        "run",
        "--rm",
        service,
        "python",
        "-m",
        "training",
        subcommand,
        *phase_args,
    ]
    proc = subprocess.run(cmd, cwd=serving.REPO_ROOT)
    if proc.returncode != 0:
        raise EvalRunError(
            f"Container phase failed ({proc.returncode}): "
            f"python -m training {subcommand}"
        )


def load_judgments(path: str) -> dict[tuple, dict]:
    """Existing judgment records keyed by (prompt_id, first side)."""
    from training.eval.generate import load_resume_jsonl

    return {
        (record["prompt_id"], record["first"]): record
        for record in load_resume_jsonl(path)
    }


def run_judgments(
    complete,
    prompts_by_id: dict[str, dict],
    responses: dict[str, dict[str, dict]],
    out_path: str,
    progress=None,
) -> list[dict]:
    """Judge every position-swapped pair; resumable and error-preserving.

    complete(messages) -> judge text (raises serving.ServingError on request
    failure). responses maps side ("baseline"/"tuned") to per-prompt-id
    response records. Pairs whose either side errored during generation are
    recorded as error outcomes, never silently dropped.
    """
    pairs = judge_mod.build_pairs(list(prompts_by_id))
    existing = load_judgments(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    results = []
    with open(out_path, "a") as f:
        for pair in pairs:
            key = (pair["prompt_id"], pair["first"])
            prior = existing.get(key)
            if prior is not None and prior["outcome"] != "error":
                results.append(prior)
                continue
            record = dict(pair)
            first = responses[pair["first"]].get(pair["prompt_id"], {})
            second = responses[pair["second"]].get(pair["prompt_id"], {})
            if "response" not in first or "response" not in second:
                record.update(
                    outcome="error", error="generation missing or failed for a side"
                )
            else:
                messages = judge_mod.judge_messages(
                    prompts_by_id[pair["prompt_id"]]["prompt"],
                    first["response"],
                    second["response"],
                )
                try:
                    text = complete(messages)
                    verdict = judge_mod.parse_verdict(text)
                    record.update(
                        outcome=judge_mod.resolve_pair(pair, verdict),
                        verdict=verdict,
                        rationale=text,
                    )
                except serving.ServingError as exc:
                    record.update(outcome="error", error=str(exc))
            f.write(json.dumps(record) + "\n")
            f.flush()
            results.append(record)
            if progress is not None:
                progress(record)
    return results


def holdout_section(base_record: dict, tuned_record: dict) -> dict:
    """Scorecard holdout section from the two container-phase records."""
    from training.eval.scorecard import ScorecardError, delta_section

    if base_record.get("precision") != tuned_record.get("precision"):
        raise ScorecardError(
            "Confound: held-out precision differs between sides "
            f"(baseline={base_record.get('precision')}, "
            f"tuned={tuned_record.get('precision')})"
        )
    # A resumed side record from a run with different parameters would make
    # an apples-to-oranges scorecard; refuse like the precision confound
    if base_record.get("dataset") != tuned_record.get("dataset"):
        raise ScorecardError(
            "Confound: held-out dataset differs between sides "
            f"(baseline={base_record.get('dataset')}, "
            f"tuned={tuned_record.get('dataset')}); delete the stale side "
            "record from the run dir and rerun"
        )
    if base_record.get("examples") != tuned_record.get("examples"):
        raise ScorecardError(
            "Confound: held-out example count differs between sides "
            f"(baseline={base_record.get('examples')}, "
            f"tuned={tuned_record.get('examples')}); delete the stale side "
            "record from the run dir and rerun"
        )
    return {
        "loss": delta_section(
            base_record["loss"], tuned_record["loss"], higher_is_better=False
        ),
        "perplexity": delta_section(
            base_record["perplexity"],
            tuned_record["perplexity"],
            higher_is_better=False,
        ),
        "examples": tuned_record["examples"],
        "supervised_tokens": tuned_record["supervised_tokens"],
        "precision": tuned_record.get("precision"),
        "attention_implementation": tuned_record.get("attention_implementation"),
        "dataset": tuned_record.get("dataset"),
    }


def lmeval_section(base_record: dict, tuned_record: dict) -> dict:
    """Scorecard lmeval section from the two container-phase records."""
    from training.eval.scorecard import ScorecardError, delta_section

    if base_record.get("lm_eval_version") != tuned_record.get("lm_eval_version"):
        raise ScorecardError("Confound: lm-eval version differs between sides")
    # Same guard as the version confound: a side resumed from a run with a
    # different --limit is not comparable
    if base_record.get("limit") != tuned_record.get("limit"):
        raise ScorecardError(
            "Confound: lm-eval limit differs between sides "
            f"(baseline={base_record.get('limit')}, "
            f"tuned={tuned_record.get('limit')}); delete the stale side "
            "record from the run dir and rerun"
        )
    tasks = {}
    for task, tuned_result in tuned_record["results"].items():
        base_result = base_record["results"].get(task)
        if base_result is None:
            raise ScorecardError(f"Task {task} missing from the baseline run")
        section = delta_section(
            base_result["value"], tuned_result["value"], higher_is_better=True
        )
        section["metric"] = tuned_result["metric"]
        section["stderr"] = {
            "baseline": base_result.get("stderr"),
            "tuned": tuned_result.get("stderr"),
        }
        tasks[task] = section
    return {
        "lm_eval_version": tuned_record.get("lm_eval_version"),
        "limit": tuned_record.get("limit"),
        "tasks": tasks,
    }
