"""`bench noise` end to end, with the host replaced.

The noise floor is the mechanism every `within_noise` verdict rests on, and until this
module existed the command that measures it was executed by no test -- which is how a
required argument added to `runner.assemble` reached the tree with this, its second call
site, left uncalled. The suite could not tell, because nothing here ran.

Everything host-side is substituted, as in test_run_command.py. No number here is real.
"""

from __future__ import annotations

from bench import delta, report, runner
from bench.cli import build_parser
from bench.commands import noise as noise_cmd
from common.serving import ServingError

MODEL = "qwen3-coder-30b-a3b-IQ4_XS"


def argv(out, *extra):
    return build_parser().parse_args(
        ["noise", "--service", "llama-cpu", "--model", MODEL, "--out", str(out), *extra]
    )


class TestTheFloorIsWritten:
    def test_a_clean_pair_writes_a_floor_document(self, host, tmp_path):
        assert noise_cmd.cmd(argv(tmp_path)) == 0
        written = list(tmp_path.glob("noise-*/noise.json"))
        assert len(written) == 1
        document = report.read(str(written[0]), expect=report.NOISE_REPORT_TYPE)
        assert document["floor"]
        assert document["run"]["service"] == "llama-cpu"

    def test_both_arms_are_assembled(self, host, tmp_path, monkeypatch):
        """The two arms go through runner.assemble, which the floor is differenced from.

        Asserted by call count rather than by output: the defect this pins is a call
        site that raises before it can produce anything, so a test reading the result
        would report the wrong cause.
        """
        seen = []
        original = runner.assemble

        def spy(**kwargs):
            seen.append(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(noise_cmd.runner, "assemble", spy)
        assert noise_cmd.cmd(argv(tmp_path)) == 0
        assert [call["arm"] for call in seen] == ["a", "a-prime"]

    def test_the_floor_records_that_its_arms_were_not_interleaved(
        self, host, tmp_path, monkeypatch
    ):
        """The two arms run back to back in one process, so the report must say so.

        The command's own description and module docstring have called them
        interleaved; they are not, and a floor that claims an ordering it did not run
        misrepresents how it was measured.
        """
        seen = []
        original = runner.assemble

        def spy(**kwargs):
            seen.append(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(noise_cmd.runner, "assemble", spy)
        noise_cmd.cmd(argv(tmp_path))
        assert {call["arm_sequence"] for call in seen} == {"blocked"}


class FailingArm:
    """A service that answers one arm and refuses part of the other.

    The floor is a difference between two arms, so a cell lost in one of them removes
    a metric from the difference rather than from one side of it. That is the asymmetry
    BQ-F44 is about, and it cannot be built from the clean fake.
    """

    def __init__(self, engine, *, fail_run_key: str, fail_points: tuple[str, ...]):
        self.engine = engine
        self.fail_run_key = fail_run_key
        self.fail_points = fail_points

    def __call__(self, adapter, body, timeout):
        # Selected off the nonce, which leads the prompt and carries the pass key and
        # the point key: "<seed>-<run key>-<point>/<gen>/<conc>-<repeat>-...".
        prompt = body["prompt"]
        if f"-{self.fail_run_key}-" in prompt and any(
            f"/{point}/" in prompt for point in self.fail_points
        ):
            raise ServingError("llama-cpu: connection reset")
        return self.engine(adapter, body, timeout)


class TestTheFloorCarriesItsOwnEvidence:
    """BQ-F44: what the pair actually did must survive into the artifact.

    The floor document was assembled by hand from a header plus four fields, so an arm
    that lost most of its cells produced a narrower floor with a smaller worst spread
    and nothing in the file saying which cells were missing, or under what conditions,
    or that anything had failed at all.
    """

    def test_a_pair_that_lost_cells_says_so(self, host, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "bench.client.stream",
            FailingArm(
                host.engine, fail_run_key="a-prime", fail_points=("p128", "p512")
            ),
        )
        assert noise_cmd.cmd(argv(tmp_path)) == 0
        document = report.read(
            str(next(tmp_path.glob("noise-*/noise.json"))),
            expect=report.NOISE_REPORT_TYPE,
        )
        assert document["errors"], "a floor built over failed cells recorded none"
        assert {e["arm"] for e in document["errors"]} == {"a-prime"}

    def test_the_conditions_the_floor_was_measured_under_are_recorded(
        self, host, tmp_path
    ):
        """A floor is only the spread of one machine in one state.

        Without the conditions block the artifact cannot be checked against the arms it
        is later applied to, which is the whole basis for refusing a mismatched floor.
        """
        assert noise_cmd.cmd(argv(tmp_path)) == 0
        document = report.read(
            str(next(tmp_path.glob("noise-*/noise.json"))),
            expect=report.NOISE_REPORT_TYPE,
        )
        assert document["conditions"]["service"] == "llama-cpu"
        assert document["conditions"]["model"]["quantization"]

    def test_the_pair_is_recoverable_from_the_artifact(self, host, tmp_path):
        """Both arms' reductions are carried, so the difference can be recomputed.

        A floor that cannot be re-derived from its own file is a number to be trusted
        rather than checked, and `worst_pct` is quoted as an acceptance signal.
        """
        assert noise_cmd.cmd(argv(tmp_path)) == 0
        document = report.read(
            str(next(tmp_path.glob("noise-*/noise.json"))),
            expect=report.NOISE_REPORT_TYPE,
        )
        assert [arm["run"]["arm"] for arm in document["arms"]] == ["a", "a-prime"]
        recomputed = delta.floor_from_pair(document["arms"][0], document["arms"][1])
        assert recomputed["floor"] == document["floor"]


class TestEachArmIsGated:
    def test_the_health_gate_runs_once_per_arm(self, host, tmp_path):
        """A service degrading between the arms must not pass unnoticed.

        Gating once for the pair checks the state the first arm ran under and assumes
        it for the second. The second arm is the one drift lands on, so it is the arm
        the assumption is least safe for -- and a floor widened by a degrading service
        suppresses findings for every comparison that later cites it.
        """
        assert noise_cmd.cmd(argv(tmp_path)) == 0
        assert host.gated == ["llama-cpu", "llama-cpu"]


class TestTheFloorNamesTheWorkloadItMeasured:
    def test_the_digest_travels_with_the_floor(self, host, tmp_path):
        """The name is what the file calls itself; the digest is what ran.

        `bench compare` refuses a floor whose digest does not match the arms', which
        only works if both sides write it.
        """
        from bench.workloads import load as load_profile

        assert noise_cmd.cmd(argv(tmp_path)) == 0
        document = report.read(
            str(next(tmp_path.glob("noise-*/noise.json"))),
            expect=report.NOISE_REPORT_TYPE,
        )
        assert (
            document["run"]["workload_hash"]
            == load_profile("quick", "llama-cpu").workload_hash
        )
