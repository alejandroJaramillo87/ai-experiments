"""What `bench compare` asserts before it prints a clean bill.

The command's exit code is what `docs/verification/bench/bench.md` B6 reads as its
acceptance signal, so every way of reaching 0 has to mean "these metrics were graded
and none regressed". Three ways of reaching it did not: an empty comparison, a
comparison whose exit code was decided by a display filter, and a noise floor that was
not a noise floor. The reads that open the command are exercised here too, because a
traceback is not a diagnosed error.
"""

from __future__ import annotations

import copy
import json

import pytest

from bench import report
from bench.cli import build_parser
from bench.commands import compare
from common.reports import report_header
from tests.bench import support


def noise_document(floor: dict) -> dict:
    """A `bench_noise` document of the shape `bench noise` writes."""
    document = report_header(
        report.NOISE_REPORT_TYPE, schema_version=report.SCHEMA_VERSION
    )
    document.update(
        git_sha="deadbeef",
        run={"pair_id": "pair", "profile": "quick", "service": "llama-cpu"},
        floor=floor,
        worst_pct=max(floor.values(), default=0.0),
    )
    return document


def set_metric(document: dict, point: str, metric: str, median: float) -> dict:
    """Attach one statistic block to a named point, in place."""
    for body in document["phases"].values():
        for entry in body.get("points", []):
            if entry["point"] == point:
                entry["metrics"][metric] = support.stat(median)
    return document


def write(tmp_path, name: str, document: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def run_compare(tmp_path, baseline, candidate, noise=None, *flags) -> int:
    argv = [
        "compare",
        write(tmp_path, "baseline.json", baseline),
        write(tmp_path, "candidate.json", candidate),
    ]
    if noise is not None:
        argv += ["--noise", write(tmp_path, "noise.json", noise)]
    argv += list(flags)
    return compare.cmd(build_parser().parse_args(argv))


DECODE_POINT = "decode/p128/g256/c1"


class TestNothingWasCompared:
    def test_a_baseline_missing_a_phase_is_not_a_clean_bill(self, tmp_path, capsys):
        # runner.reduce drops a phase whose cells all failed, and check_comparable
        # never inspects phases, so the pair passes as comparable. The candidate is
        # 50% slower on decode and the decode line is never printed.
        baseline = support.result(decode_tok_s=40.0)
        del baseline["phases"]["decode"]
        candidate = support.result(decode_tok_s=20.0)

        code = run_compare(tmp_path, baseline, candidate)
        err = capsys.readouterr().err
        assert code != 0
        assert "no regression" not in err

    def test_a_pair_sharing_no_metric_is_not_a_clean_bill(self, tmp_path, capsys):
        baseline = support.result()
        candidate = support.result()
        baseline["phases"] = {}
        candidate["phases"] = {}

        code = run_compare(tmp_path, baseline, candidate)
        err = capsys.readouterr().err
        assert code == 1
        assert "no regression" not in err
        assert "[ERROR]" in err

    def test_a_failed_cell_recorded_in_an_arm_is_surfaced(self, tmp_path, capsys):
        candidate = support.result()
        candidate["errors"] = [{"point": DECODE_POINT, "error": "read timeout"}]

        run_compare(tmp_path, support.result(), candidate)
        err = capsys.readouterr().err
        assert "[WARN]" in err
        assert "candidate" in err

    def test_a_complete_pair_still_passes(self, tmp_path, capsys):
        code = run_compare(tmp_path, support.result(), support.result())
        assert code == 0
        assert "no regression" in capsys.readouterr().err


def _itl_pair() -> tuple[dict, dict, dict]:
    """A pair whose only regression is on a metric outside the display set."""
    baseline = set_metric(support.result(), DECODE_POINT, "itl_ms", 10.0)
    candidate = set_metric(support.result(), DECODE_POINT, "itl_ms", 14.0)
    floor = noise_document({f"{DECODE_POINT}:itl_ms": 5.0})
    return baseline, candidate, floor


class TestTheExitCodeIsNotADisplayDecision:
    def test_a_regression_outside_the_display_set_still_fails(self, tmp_path):
        baseline, candidate, floor = _itl_pair()
        assert run_compare(tmp_path, baseline, candidate, floor) == 1

    def test_the_display_flag_does_not_move_the_exit_code(self, tmp_path):
        baseline, candidate, floor = _itl_pair()
        plain = run_compare(tmp_path, baseline, candidate, floor)
        verbose = run_compare(tmp_path, baseline, candidate, floor, "--all")
        assert plain == verbose == 1

    def test_the_failing_metric_is_named_even_when_it_is_not_tabulated(
        self, tmp_path, capsys
    ):
        baseline, candidate, floor = _itl_pair()
        run_compare(tmp_path, baseline, candidate, floor)
        assert "itl_ms" in capsys.readouterr().err

    def test_a_diagnostic_count_is_not_graded_as_a_performance_metric(self, tmp_path):
        # prompt_tokens_observed records what the server tokenized. A different
        # number is a workload difference to investigate, not a slower engine.
        baseline = set_metric(
            support.result(), DECODE_POINT, "prompt_tokens_observed", 128.0
        )
        candidate = set_metric(
            support.result(), DECODE_POINT, "prompt_tokens_observed", 192.0
        )
        floor = noise_document({f"{DECODE_POINT}:prompt_tokens_observed": 5.0})
        assert run_compare(tmp_path, baseline, candidate, floor, "--all") == 0

    def test_the_json_payload_accounts_for_every_regression_it_reports(
        self, tmp_path, capsys
    ):
        baseline, candidate, floor = _itl_pair()
        code = run_compare(tmp_path, baseline, candidate, floor, "--json")
        payload = json.loads(capsys.readouterr().out)
        assert code == 1
        assert payload["regressions"] == 1
        graded = [d for d in payload["deltas"] if d["verdict"] == "regression"]
        assert [d["metric"] for d in graded] == ["itl_ms"]


class TestTheNoiseFloorMustBeANoiseFloor:
    def test_a_result_file_is_refused_as_a_floor(self, tmp_path, capsys):
        code = run_compare(
            tmp_path, support.result(), support.result(), support.result()
        )
        err = capsys.readouterr().err
        assert code == 2
        assert report.NOISE_REPORT_TYPE in err

    def test_a_floor_is_refused_as_an_arm(self, tmp_path, capsys):
        floor = noise_document({f"{DECODE_POINT}:itl_ms": 5.0})
        code = run_compare(tmp_path, floor, support.result())
        err = capsys.readouterr().err
        assert code == 2
        assert report.REPORT_TYPE in err


class TestUnreadableInputsAreDiagnosed:
    def test_a_missing_file_is_an_error_not_a_traceback(self, tmp_path, capsys):
        argv = ["compare", str(tmp_path / "absent.json"), str(tmp_path / "b.json")]
        code = compare.cmd(build_parser().parse_args(argv))
        err = capsys.readouterr().err
        assert code == 2
        assert "[ERROR]" in err
        assert "absent.json" in err

    def test_a_truncated_file_is_an_error_not_a_traceback(self, tmp_path, capsys):
        half = tmp_path / "half.json"
        half.write_text(json.dumps(support.result())[:200], encoding="utf-8")
        argv = [
            "compare",
            str(half),
            write(tmp_path, "candidate.json", support.result()),
        ]
        code = compare.cmd(build_parser().parse_args(argv))
        assert code == 2
        assert "[ERROR]" in capsys.readouterr().err

    def test_a_document_failing_the_envelope_is_an_error_not_a_traceback(
        self, tmp_path, capsys
    ):
        broken = copy.deepcopy(support.result())
        del broken["schema_version"]
        argv = [
            "compare",
            write(tmp_path, "candidate.json", support.result()),
            write(tmp_path, "broken.json", broken),
        ]
        code = compare.cmd(build_parser().parse_args(argv))
        assert code == 2
        assert "[ERROR]" in capsys.readouterr().err

    @pytest.mark.parametrize("flags", [(), ("--json",)])
    def test_no_reader_failure_escapes_the_command(self, tmp_path, flags):
        argv = ["compare", str(tmp_path / "absent.json"), str(tmp_path / "b.json")]
        argv += list(flags)
        assert compare.cmd(build_parser().parse_args(argv)) == 2
