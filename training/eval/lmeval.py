"""lm-evaluation-harness evaluator (Phase 1 F7).

Runs the pinned task set (training/eval/tasks.yaml) in-process against the
HF checkpoint — before quantization, so the numbers measure the fine-tune,
not the quant. Adapter-only checkpoints load through the harness's peft
support against their recorded base. Harness and task versions are recorded
in the result so the scorecard metadata can pin them.

lm_eval imports lazily inside the training container; task-file parsing is
importable anywhere for unit tests.
"""

from __future__ import annotations

import os

import yaml

DEFAULT_TASKS_FILE = os.path.join(os.path.dirname(__file__), "tasks.yaml")


class LmEvalError(RuntimeError):
    """Harness failure or task-set/config mismatch; message says the fix."""


def load_task_config(tasks_file: str = DEFAULT_TASKS_FILE,
                     tasks: list[str] | None = None) -> dict:
    """{task: {"metric": ...}} from tasks.yaml, optionally filtered.

    An explicitly requested task missing from the pinned set is an error:
    every scored task must have a declared metric, or the scorecard could
    not extract a comparable number.
    """
    with open(tasks_file) as f:
        config = yaml.safe_load(f)
    declared = config.get("tasks") or {}
    if tasks is None:
        return {"lm_eval_version": config.get("lm_eval_version"),
                "tasks": declared}
    unknown = [t for t in tasks if t not in declared]
    if unknown:
        raise LmEvalError(
            f"Tasks not in the pinned set ({tasks_file}): {', '.join(unknown)}\n"
            "  Solution: add the task and its metric key to tasks.yaml")
    return {"lm_eval_version": config.get("lm_eval_version"),
            "tasks": {t: declared[t] for t in tasks}}


def extract_results(raw: dict, task_config: dict) -> dict:
    """Per-task {metric, value, stderr} from a harness results payload."""
    out = {}
    for task, spec in task_config["tasks"].items():
        metric = spec["metric"]
        task_results = raw.get("results", {}).get(task)
        if task_results is None:
            raise LmEvalError(f"Task missing from harness results: {task}")
        if metric not in task_results:
            raise LmEvalError(
                f"Metric {metric} missing from {task} results "
                f"(available: {', '.join(sorted(task_results))})\n"
                "  Solution: fix the metric key in tasks.yaml")
        stderr_key = metric.split(",")[0] + "_stderr" + (
            "," + metric.split(",", 1)[1] if "," in metric else "")
        out[task] = {
            "metric": metric,
            "value": task_results[metric],
            "stderr": task_results.get(stderr_key),
            "version": raw.get("versions", {}).get(task),
        }
    return out


def run_lmeval(model: str, tasks_file: str = DEFAULT_TASKS_FILE,
               tasks: list[str] | None = None,
               limit: int | None = None,
               batch_size: str = "auto") -> dict:
    """Harness run for one model; returns the scorecard-ready record."""
    import lm_eval

    from training.eval.holdout import resolve_model

    task_config = load_task_config(tasks_file, tasks)
    pinned = task_config.get("lm_eval_version")
    if pinned and lm_eval.__version__ != pinned:
        raise LmEvalError(
            f"lm-eval version mismatch: installed {lm_eval.__version__}, "
            f"tasks.yaml pins {pinned}\n"
            "  Solution: rebuild the training image or update the pin "
            "deliberately (one bump at a time)")

    base, adapter = resolve_model(model)
    model_args = f"pretrained={base},dtype=bfloat16"
    if adapter is not None:
        model_args += f",peft={adapter}"
    raw = lm_eval.simple_evaluate(
        model="hf", model_args=model_args,
        tasks=list(task_config["tasks"]), limit=limit,
        batch_size=batch_size)
    return {
        "model": model,
        "lm_eval_version": lm_eval.__version__,
        "tasks_file": os.path.abspath(tasks_file),
        "limit": limit,
        "results": extract_results(raw, task_config),
    }
