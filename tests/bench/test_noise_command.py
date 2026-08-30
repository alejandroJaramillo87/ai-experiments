"""`bench noise` end to end, with the host replaced.

The noise floor is the mechanism every `within_noise` verdict rests on, and until this
module existed the command that measures it was executed by no test -- which is how a
required argument added to `runner.assemble` reached the tree with this, its second call
site, left uncalled. The suite could not tell, because nothing here ran.

Everything host-side is substituted, as in test_run_command.py. No number here is real.
"""

from __future__ import annotations

from bench import report, runner
from bench.cli import build_parser
from bench.commands import noise as noise_cmd

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
