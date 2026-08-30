"""`bench run` end to end, with the host replaced.

The command is the part that owns the run lifecycle: which arm runs when, what is
recreated between them, what reaches disk, and what a failure leaves behind. None of
that needs a container, so everything host-side -- compose, the health poll, the wire,
and the conditions probes that shell out to docker and nvidia-smi -- is substituted
here, and the test asserts on the order of what was called and on what was written.

What cannot be proven here is any actual number.
"""

from __future__ import annotations

import pytest

from bench.cli import build_parser
from bench.commands import run as run_cmd
from bench.report import read

MODEL = "qwen3-coder-30b-a3b-IQ4_XS"


def argv(*extra, out=None):
    args = ["run", "--service", "llama-cpu", "--model", MODEL, *extra]
    if out is not None:
        args += ["--out", str(out)]
    return build_parser().parse_args(args)


class TestOutputRoot:
    """--out is knowable before the service is touched, so it is checked there.

    The default is /mnt/ai-data/benchmarks/bench, which exists on the workstation and
    nowhere else. Discovering that after a recreate, a health gate and an hour of
    workload throws away the whole measurement over a typo.
    """

    def test_an_unusable_output_root_is_refused_before_anything_runs(self, host):
        args = argv(out="/proc/version/bench")
        assert run_cmd.cmd(args) == 2
        assert host.gated == [], "the health gate ran against an unusable --out"
        assert host.composed == []
        assert host.engine.prompts == [], "a workload ran with nowhere to write it"

    def test_a_usable_root_is_accepted(self, host, tmp_path):
        assert run_cmd.cmd(argv(out=tmp_path / "results")) == 0


class TestArmOrdering:
    """Whether the arms alternate, and whether the report says which happened.

    Running one arm's whole profile before the other's lets any drift over the run --
    thermal, a background process, a clock that settles -- land entirely on the second
    arm and be read as the arm's effect. Interleaving at the repeat level spreads it
    across both. It is not free: every repeat costs a container recreate and a full
    warm-up pass per arm, which is why it is opt-in rather than the default.
    """

    def test_blocked_is_the_default_and_the_report_says_so(self, host, tmp_path):
        args = argv(
            "--arm", "on:HUGEPAGES=on", "--arm", "off:HUGEPAGES=off", out=tmp_path
        )
        assert run_cmd.cmd(args) == 0
        assert host.composed == [{"HUGEPAGES": "on"}, {"HUGEPAGES": "off"}]
        for path in sorted(tmp_path.glob("*/result.json")):
            assert read(str(path))["run"]["arm_sequence"] == "blocked"

    def test_interleave_alternates_the_arms_at_every_repeat(self, host, tmp_path):
        args = argv(
            "--arm",
            "on:HUGEPAGES=on",
            "--arm",
            "off:HUGEPAGES=off",
            "--interleave",
            out=tmp_path,
        )
        assert run_cmd.cmd(args) == 0
        # quick.toml is three repeats, so each arm is recreated and gated three times,
        # alternating -- not once each, back to back.
        assert host.composed == [{"HUGEPAGES": "on"}, {"HUGEPAGES": "off"}] * 3
        assert host.gated == ["llama-cpu"] * 6

    def test_an_interleaved_report_says_it_was_interleaved(self, host, tmp_path):
        args = argv(
            "--arm",
            "on:HUGEPAGES=on",
            "--arm",
            "off:HUGEPAGES=off",
            "--interleave",
            out=tmp_path,
        )
        assert run_cmd.cmd(args) == 0
        written = sorted(tmp_path.glob("*/result.json"))
        assert len(written) == 2
        for path in written:
            assert read(str(path))["run"]["arm_sequence"] == "interleaved"

    def test_an_interleaved_run_keeps_every_repeat_of_both_arms(self, host, tmp_path):
        args = argv(
            "--arm",
            "on:HUGEPAGES=on",
            "--arm",
            "off:HUGEPAGES=off",
            "--interleave",
            out=tmp_path,
        )
        assert run_cmd.cmd(args) == 0
        for path in sorted(tmp_path.glob("*/result.json")):
            document = read(str(path))
            for phase in document["phases"].values():
                for point in phase["points"]:
                    assert point["metrics"]["ttft_ms"]["n"] == 3

    def test_an_interleaved_run_repeats_no_prompt(self, host, tmp_path):
        # Every pass re-sends the warm-up cells, so the nonce has to separate passes as
        # well as arms: two passes sending identical bytes measure a prefix cache.
        args = argv(
            "--arm",
            "on:HUGEPAGES=on",
            "--arm",
            "off:HUGEPAGES=off",
            "--interleave",
            out=tmp_path,
        )
        assert run_cmd.cmd(args) == 0
        prompts = host.engine.prompts
        assert len(set(prompts)) == len(prompts)


BLOCKED_PROFILE = """\
name    = "blocked-fixture"
repeats = 2
warmup  = 1
order   = "blocked"
seed    = 42

[prefill]
prompt_tokens = [128, 1024]
max_tokens = 1

[decode]
prompt_tokens = 128
max_tokens = [64]

[concurrency]
llama-cpu = [1]
"""


@pytest.fixture
def blocked_profile(tmp_path):
    """A profile whose own point ordering is blocked, written to disk.

    `load` takes a path as well as a name, so the combination can be exercised without
    committing a profile the protocol forbids for real use.
    """
    path = tmp_path / "blocked.toml"
    path.write_text(BLOCKED_PROFILE, encoding="utf-8")
    return str(path)


class TestInterleaveRefusals:
    """The two combinations whose report would describe a run that never happened."""

    def test_interleaving_a_single_arm_is_refused(self, host, tmp_path):
        args = argv("--interleave", out=tmp_path)
        assert run_cmd.cmd(args) == 2
        assert host.composed == []
        assert host.engine.prompts == []

    def test_interleaving_a_blocked_profile_is_refused(
        self, host, tmp_path, blocked_profile
    ):
        # Partitioning by repeat runs the points round-robin, so the profile's own
        # blocked order would be false in the report that recorded it.
        args = argv(
            "--profile",
            blocked_profile,
            "--arm",
            "on:HUGEPAGES=on",
            "--arm",
            "off:HUGEPAGES=off",
            "--interleave",
            out=tmp_path / "results",
        )
        assert run_cmd.cmd(args) == 2
        assert host.engine.prompts == [], "a workload ran under a refused combination"

    def test_a_blocked_profile_still_runs_without_the_flag(
        self, host, tmp_path, blocked_profile
    ):
        args = argv("--profile", blocked_profile, out=tmp_path / "results")
        assert run_cmd.cmd(args) == 0
        written = sorted((tmp_path / "results").glob("*/result.json"))
        document = read(str(written[0]))
        assert document["run"]["order"] == "blocked"
        assert document["run"]["arm_sequence"] == "blocked"
