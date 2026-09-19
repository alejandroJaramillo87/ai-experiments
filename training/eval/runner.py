"""Host-side phase orchestration for the F7 runner.

The runner (training/commands/eval_ft.py) runs the two in-process
evaluators (holdout, lmeval) inside the training container with inference
down. This module holds the testable pieces — checkpoint identity,
container-phase invocation, section assembly — with the container call
delegated to docker compose.

Identity: a compared side is an HF checkpoint directory and is identified by
model_hash, the sha256 of its model.safetensors, read from the export
manifest beside it when there is one and computed otherwise.

Host-side, stdlib-only.
"""

from __future__ import annotations

import json
import os
import subprocess

from training import serving

TRAIN_COMPOSE = serving.TRAIN_COMPOSE
DEFAULT_RUN_ROOT = "/mnt/ai-data/training/eval"


class EvalRunError(RuntimeError):
    """Runner failure outside the serving layer; message says the fix."""


def thinking_mode_for(model_type: str | None) -> str | None:
    """The thinking mode a served checkpoint actually runs in.

    No thinking-control tokens are injected anywhere in the eval pipeline,
    so the mode is the family default (off for Gemma 4, whose <|think|> is
    opt-in). Family comes from the checkpoint's config.json model_type; the
    composite Gemma 4 config says gemma4 and its text-only form says
    gemma4_text, and both resolve to the gemma4 family. None when the type
    names no registered family (recorded as "not recorded", never guessed).
    """
    from training.models.families import family_spec, supported_families

    if not model_type:
        return None
    for family in supported_families():
        if model_type == family or model_type.startswith(family + "_"):
            return family_spec(family).thinking_default
    return None


def hf_identity(model: str) -> dict:
    """Identity block for one compared side: the checkpoint directory the
    harness serves, and its model_hash.

    The directory must live under the models tree the harness mounts, in
    either its host or container spelling, and hold config.json and
    model.safetensors. model_hash comes from the export manifest beside it
    when there is one (an exported fine-tune), else is computed from the
    weights (a base checkpoint), by the same function either way.
    """
    from training.export.errors import ExportError
    from training.export.manifest import load_hf_manifest

    try:
        container_path, host_path = serving.model_paths(model.rstrip("/"))
    except serving.ServingError as exc:
        raise EvalRunError(
            f"{exc}\n  Solution: the harness serves checkpoints under "
            f"{serving.HOST_MODELS}; pass the exported hf directory or the "
            "downloaded base"
        ) from exc
    config_path = os.path.join(host_path, "config.json")
    weights_path = os.path.join(host_path, "model.safetensors")
    for path in (config_path, weights_path):
        if not os.path.isfile(path):
            raise EvalRunError(
                f"Not a loadable checkpoint directory: {host_path} "
                f"(missing {os.path.basename(path)})\n  Solution: pass the "
                "hf/ directory python -m training export wrote, or the "
                "downloaded base checkpoint"
            )
    try:
        manifest = load_hf_manifest(host_path)
        model_hash = manifest["model_hash"]
        manifest_path = os.path.join(os.path.dirname(host_path), "manifest.json")
    except ExportError:
        model_hash = serving.sha256_file(weights_path)
        manifest_path = None
    with open(config_path) as f:
        model_type = json.load(f).get("model_type")
    return {
        "model": container_path,
        "host_path": host_path,
        "model_hash": model_hash,
        "manifest": manifest_path,
        "thinking_mode": thinking_mode_for(model_type),
    }


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
