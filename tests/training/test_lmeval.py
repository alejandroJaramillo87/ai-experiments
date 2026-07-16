"""lm-eval evaluator helpers: result extraction, the stderr-key composition
for filtered metrics, and the version-mismatch guard. lm_eval is stubbed via
sys.modules so the guard is exercised without installing the harness."""

import sys
import types

import pytest

from training.eval.lmeval import LmEvalError, extract_results, run_lmeval


def test_extract_results_happy_path_and_stderr_keys():
    raw = {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.42,
                "exact_match_stderr,strict-match": 0.05,
            },
            "tinyMMLU": {"acc_norm": 0.70, "acc_norm_stderr": 0.03},
        },
        "versions": {"gsm8k": 3, "tinyMMLU": 0},
    }
    task_config = {
        "tasks": {
            "gsm8k": {"metric": "exact_match,strict-match"},
            "tinyMMLU": {"metric": "acc_norm"},
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
    # Plain metric: the stderr key is simply <metric>_stderr.
    assert out["tinyMMLU"] == {
        "metric": "acc_norm",
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
    # The task ran but the declared metric key is absent: the scorecard could
    # not extract a comparable number, so refuse.
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
