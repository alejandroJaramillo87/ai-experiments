"""lm-eval evaluator helpers: result extraction, the stderr-key composition
for filtered metrics, the version-mismatch guard, and the repo-owned task
configs the harness's registry is extended with. lm_eval is stubbed via
sys.modules so the guard is exercised without installing the harness."""

import os
import sys
import types

import pytest
import yaml

from training.eval.lmeval import (
    REPO_TASKS_DIR,
    LmEvalError,
    extract_results,
    load_task_config,
    run_lmeval,
)


def test_extract_results_happy_path_and_stderr_keys():
    raw = {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.42,
                "exact_match_stderr,strict-match": 0.05,
            },
            "tinyMMLU_acc": {"acc_norm,none": 0.70, "acc_norm_stderr,none": 0.03},
        },
        "versions": {"gsm8k": 3, "tinyMMLU_acc": 0},
    }
    task_config = {
        "tasks": {
            "gsm8k": {"metric": "exact_match,strict-match"},
            "tinyMMLU_acc": {"metric": "acc_norm,none"},
        }
    }
    out = extract_results(raw, task_config)
    # Filtered metric: the stderr key splits the base metric off, appends
    # _stderr, then re-attaches the filter suffix -> exact_match_stderr,strict-match.
    assert out["gsm8k"] == {
        "metric": "exact_match,strict-match",
        "value": 0.42,
        "stderr": 0.05,
        "version": 3,
    }
    # Unfiltered task: the harness names its default filter none, so the
    # same composition gives acc_norm_stderr,none (KB-94).
    assert out["tinyMMLU_acc"] == {
        "metric": "acc_norm,none",
        "value": 0.70,
        "stderr": 0.03,
        "version": 0,
    }


def test_extract_results_refuses_missing_task():
    # A scored task absent from the harness payload is an error, not a skip.
    task_config = {"tasks": {"gsm8k": {"metric": "acc"}}}
    with pytest.raises(LmEvalError, match="Task missing from harness results"):
        extract_results({"results": {}}, task_config)


def test_extract_results_refuses_missing_metric():
    # The task ran but the declared metric key is absent: the record could
    # not carry a comparable number, so refuse.
    raw = {"results": {"gsm8k": {"acc_norm": 0.5}}}
    task_config = {"tasks": {"gsm8k": {"metric": "exact_match"}}}
    with pytest.raises(LmEvalError, match="Metric exact_match missing"):
        extract_results(raw, task_config)


def test_run_lmeval_refuses_version_mismatch(monkeypatch):
    # The pinned set (tasks.yaml) declares lm_eval 0.4.12; a harness reporting
    # a different __version__ must refuse before any evaluation runs.
    fake = types.ModuleType("lm_eval")
    fake.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "lm_eval", fake)
    with pytest.raises(LmEvalError, match="version mismatch"):
        run_lmeval("org/does-not-exist")


def test_repo_task_configs_are_plain_yaml_and_pinned():
    # Every repo-owned task (KB-93) parses with safe_load, so it carries no
    # !function tag reaching into a package the image may not have, and the
    # pinned set names each one by the task name the file declares.
    files = sorted(f for f in os.listdir(REPO_TASKS_DIR) if f.endswith(".yaml"))
    assert files == ["tinyMMLU_acc.yaml"]
    pinned = load_task_config()["tasks"]
    for name in files:
        with open(os.path.join(REPO_TASKS_DIR, name)) as f:
            task = yaml.safe_load(f)
        assert task["task"] == name[: -len(".yaml")]
        assert task["task"] in pinned
        # The task file names bare metrics; the pinned key carries the filter.
        assert pinned[task["task"]]["metric"].split(",")[0] in {
            m["metric"] for m in task["metric_list"]
        }


def test_pinned_metric_keys_carry_the_filter_suffix():
    # lm-eval 0.4.12 keys every aggregated result as <metric>,<filter>, the
    # filter being "none" for a task without a filter block, so a bare key in
    # tasks.yaml can never match a result (KB-94: the first eval-lmeval to
    # reach scoring refused ifeval on prompt_level_strict_acc).
    for task, spec in load_task_config()["tasks"].items():
        metric, _, filter_name = spec["metric"].partition(",")
        assert metric and filter_name, f"{task}: {spec['metric']} names no filter"


def test_tinymmlu_acc_is_tinymmlu_with_the_mean():
    # The same 100 items as the harness's tinyMMLU, scored as accuracy: the IRT
    # aggregation needed a git-only package and all 100 items per run.
    with open(os.path.join(REPO_TASKS_DIR, "tinyMMLU_acc.yaml")) as f:
        task = yaml.safe_load(f)
    assert (task["dataset_path"], task["dataset_name"]) == (
        "tinyBenchmarks/tinyMMLU",
        "all",
    )
    assert task["test_split"] == "test"
    assert task["output_type"] == "multiple_choice"
    assert task["metric_list"] == [
        {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True}
    ]
