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


class TestAPairMustBeOnePair:
    """BQ-F50: the pair identifier had writers and no reader.

    When the second arm of a pair fails, the first arm's file stays on disk carrying a
    pair identifier whose sibling does not exist -- indistinguishable from a completed
    pair. Reproduced before the fix: that orphan, compared against a different pair's
    arm of the same name, graded fifteen metrics with no warning and exited 0.
    """

    def test_two_halves_of_different_pairs_are_refused(self, tmp_path, capsys):
        baseline = support.result(arm="on", pair_id="pair-a")
        candidate = support.result(arm="off", pair_id="pair-b")
        assert run_compare(tmp_path, baseline, candidate) == 2
        assert "pair" in capsys.readouterr().err

    def test_one_half_of_a_pair_against_a_standalone_run_warns(self, tmp_path, capsys):
        baseline = support.result(arm="on", pair_id="pair-a")
        candidate = support.result(arm="off")
        run_compare(tmp_path, baseline, candidate)
        assert "sibling" in capsys.readouterr().err

    def test_the_two_arms_of_one_pair_compare_cleanly(self, tmp_path, capsys):
        baseline = support.result(arm="on", pair_id="pair-a")
        candidate = support.result(arm="off", pair_id="pair-a")
        run_compare(tmp_path, baseline, candidate)
        assert "pair" not in capsys.readouterr().err

    def test_two_standalone_runs_still_compare(self, tmp_path, capsys):
        run_compare(tmp_path, support.result(arm="on"), support.result(arm="off"))
        assert "pair" not in capsys.readouterr().err


class TestAnArmMustActuallyDifferFromItsPartner:
    """BQ-F52: two arm names over one configuration measure nothing.

    THREADS was a literal in docker-compose.inference.yaml and NO_REPACK appeared in no
    compose file, so `--arm t12:THREADS=12` set a variable that never crossed into the
    container. Both arms ran the same service and the comparison reported a clean null
    on an A/B that measured nothing. The compose file is fixed; this is the check that
    would have caught it, and the disproof was already in the artifact.
    """

    def test_identical_service_state_under_two_arm_names_is_refused(
        self, tmp_path, capsys
    ):
        knobs = {"THREADS": "16"}
        argv = ["llama-server", "--threads", "16"]
        baseline = support.result(arm="t12", env_knobs=knobs, server_argv=argv)
        candidate = support.result(arm="t16", env_knobs=knobs, server_argv=argv)
        assert run_compare(tmp_path, baseline, candidate) == 2
        assert "measured nothing" in capsys.readouterr().err

    def test_a_real_difference_is_compared(self, tmp_path):
        baseline = support.result(
            arm="t12",
            env_knobs={"THREADS": "12"},
            server_argv=["llama-server", "--threads", "12"],
        )
        candidate = support.result(
            arm="t16",
            env_knobs={"THREADS": "16"},
            server_argv=["llama-server", "--threads", "16"],
        )
        assert run_compare(tmp_path, baseline, candidate) == 0

    def test_an_unobserved_container_is_not_accused(self, tmp_path):
        """docker inspect can fail, and both arms then carry nothing to compare.

        Refusing on that would refuse every comparison run where the container was
        named differently -- an absent fact, not two identical ones.
        """
        assert (
            run_compare(tmp_path, support.result(arm="a"), support.result(arm="b")) == 0
        )


class TestAPartialRunIsNotAnArm:
    """A run that collapsed carries the cells that ran and no marker a reader used.

    Differenced against a full arm it reports the missing cells as whatever the
    survivors happened to measure, which is the failure mode the status field exists
    to prevent.
    """

    def test_a_partial_baseline_is_refused(self, tmp_path, capsys):
        baseline = support.result(arm="on", status="partial")
        assert run_compare(tmp_path, baseline, support.result(arm="off")) == 2
        assert "partial" in capsys.readouterr().err


class TestTheFloorMustMatchTheWorkloadItGrades:
    def test_a_floor_from_a_different_workload_is_refused(self, tmp_path, capsys):
        """The profile name is not the workload: a file can be edited under its name.

        `quick` and `publish` share the point key prefill/p128/g1/c1, so a stale floor
        lands on exactly the keys a regression would be graded against.
        """
        baseline = support.result(arm="on", workload_hash="aaa")
        candidate = support.result(arm="off", workload_hash="aaa")
        floor = noise_document({"decode/p128/g256/c1:decode_tok_s": 40.0})
        floor["run"]["workload_hash"] = "bbb"
        assert run_compare(tmp_path, baseline, candidate, floor) == 2
        assert "workload_hash" in capsys.readouterr().err

    def test_a_floor_measured_over_failed_cells_is_refused(self, tmp_path, capsys):
        """BQ-F44's consumer half.

        An arm that lost eight of nine cells wrote a floor of four entries where a
        clean run writes fifteen. The metrics it does not carry grade as inconclusive,
        which does not fail, so that floor returned a green all-clear on a genuine
        thirty-three percent decode regression.
        """
        floor = noise_document({"decode/p128/g256/c1:decode_tok_s": 1.0})
        floor["errors"] = [{"cell": "prefill/p128/g1/c1", "repeat": 0, "arm": "a"}]
        floor["coverage"] = {"floor_metrics": 1, "arm_metrics": {"a": 12, "a-prime": 4}}
        assert (
            run_compare(
                tmp_path, support.result(arm="on"), support.result(arm="off"), floor
            )
            == 2
        )
        assert "cell(s) failed while it was being measured" in capsys.readouterr().err


class TestTheOrderingWarningDetectsTheOrdering:
    """BQ-F24: the warning read run.order, which is the ordering within one pass.

    Whether the arms alternated is run.arm_sequence, a different axis, and the old
    condition could never be true for a profile the runner would accept.
    """

    def test_a_blocked_pair_is_warned_about(self, tmp_path, capsys):
        baseline = support.result(arm="on", arm_sequence="blocked")
        candidate = support.result(arm="off", arm_sequence="blocked")
        run_compare(tmp_path, baseline, candidate)
        assert "not run interleaved" in capsys.readouterr().err

    def test_an_interleaved_pair_is_not(self, tmp_path, capsys):
        baseline = support.result(arm="on", arm_sequence="interleaved")
        candidate = support.result(arm="off", arm_sequence="interleaved")
        run_compare(tmp_path, baseline, candidate)
        assert "not run interleaved" not in capsys.readouterr().err


class TestAPackedChunkIsNotAFailedCell:
    """BQ-F55: a metric refused at the reduction was diagnosed as a failed cell.

    When one arm's samples resolved their generation and the other's did not,
    bench/metrics.py nulls tpot_ms and decode_tok_s for the whole point in that arm
    only. The comparison enumerated the pair for one arm and not the other and exited 1
    under "rerun the arm whose cells failed". No cell failed: both errors lists are
    empty, and the cause is the server packing several tokens into one chunk, so a
    rerun reproduces it. The exit code was right; only the diagnosis was false.
    """

    def packed(self, arm: str) -> dict:
        document = support.result(arm=arm)
        for body in document["phases"].values():
            for entry in body.get("points", []):
                if entry["point"] == "decode/p128/g256/c1":
                    entry["metrics"].update(
                        arrivals_incomplete=4,
                        itl_approximate=True,
                        tokens_per_chunk_mean_max=3.4,
                        n=5,
                    )
                    entry["metrics"].pop("decode_tok_s")
        return document

    def test_the_packing_is_named_and_the_failed_cells_are_not(self, tmp_path, capsys):
        baseline = support.result(arm="on")
        assert run_compare(tmp_path, baseline, self.packed("off")) == 1
        printed = capsys.readouterr().err
        assert "packed" in printed
        assert "rerun the arm whose cells failed" not in printed

    def test_a_genuinely_failed_cell_still_says_so(self, tmp_path, capsys):
        baseline = support.result(arm="on")
        candidate = support.result(arm="off")
        for body in candidate["phases"].values():
            body["points"] = [
                e for e in body.get("points", []) if e["point"] != "decode/p128/g256/c1"
            ]
        candidate["errors"] = [
            {"cell": "decode/p128/g256/c1", "repeat": 0, "error": "connection reset"}
        ]
        assert run_compare(tmp_path, baseline, candidate) == 1
        printed = capsys.readouterr().err
        assert "failed cell" in printed
        assert "packed" not in printed


class TestTheFloorExcludesTokenCounts:
    """BQ-F25, pinned rather than assumed closed.

    floor_from_pair builds from compare's deltas, and the observed token counts moved
    out of the graded set at the reduction, so a tokenization artifact can no longer
    set the published worst spread. Nothing asserted that, and it is one edit in
    delta.DIAGNOSTIC away from coming back.
    """

    def test_an_observed_token_count_cannot_set_the_worst_spread(self):
        from bench import delta

        arm_a = support.result(arm="a")
        arm_b = support.result(arm="a")
        set_metric(arm_a, "decode/p128/g256/c1", "prompt_tokens_observed", 128.0)
        set_metric(arm_b, "decode/p128/g256/c1", "prompt_tokens_observed", 512.0)
        floor = delta.floor_from_pair(arm_a, arm_b)
        assert floor["worst_pct"] == 0.0
        assert not [key for key in floor["floor"] if "tokens_observed" in key]
