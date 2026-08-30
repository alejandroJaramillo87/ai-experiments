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


class Collapsing:
    """A service that answers a few cells and then fails in a way nothing anticipated.

    Deliberately not a ServingError: that type is caught per cell and recorded, so it
    can never lose a run. What loses a run is everything else -- a client bug, a
    protocol change, an OSError from the socket layer -- arriving forty minutes in.
    """

    def __init__(self, engine, *, after: int):
        self.engine = engine
        self.after = after
        self.calls = 0

    def __call__(self, adapter, body, timeout):
        self.calls += 1
        if self.calls > self.after:
            raise RuntimeError("chunked encoding ended without a terminating frame")
        return self.engine(adapter, body, timeout)


class TestAnUnexpectedFailureKeepsWhatWasMeasured:
    """BQ-F07: a run is up to an hour, and it used to write nothing until the end.

    Samples were buffered in memory and flushed once, and `assemble` sat outside the
    error handler, so any failure the serving error type does not cover discarded every
    cell that had already succeeded. The samples are the irreplaceable half -- the
    reduction can be recomputed from them, and the failure that lost them is usually
    the one they would have explained.
    """

    def test_the_samples_measured_before_the_failure_are_on_disk(
        self, host, tmp_path, monkeypatch
    ):
        engine = Collapsing(host.engine, after=8)
        monkeypatch.setattr("bench.client.stream", engine)
        assert run_cmd.cmd(argv(out=tmp_path)) == 1
        written = list(tmp_path.glob("*/samples.jsonl"))
        assert len(written) == 1, "an hour of measurement left nothing on disk"
        # Three of the first eight cells are warm-ups, which are discarded by design.
        assert len(written[0].read_text().splitlines()) == 5

    def test_the_partial_report_says_it_is_partial(self, host, tmp_path, monkeypatch):
        """A partial result that reads as a complete one is worse than none.

        It would be differenced against a full arm without complaint, and the missing
        cells would show up as whatever the surviving ones happened to measure.
        """
        monkeypatch.setattr("bench.client.stream", Collapsing(host.engine, after=8))
        assert run_cmd.cmd(argv(out=tmp_path)) == 1
        document = read(str(next(tmp_path.glob("*/result.json"))))
        assert document["run"]["status"] == "partial"
        assert any("terminating frame" in e["error"] for e in document["errors"])

    def test_a_completed_run_says_so(self, host, tmp_path):
        assert run_cmd.cmd(argv(out=tmp_path)) == 0
        document = read(str(next(tmp_path.glob("*/result.json"))))
        assert document["run"]["status"] == "complete"


class TestBothArmsAreRecreated:
    """BQ-F51: an arm with no overrides used to skip the recreate entirely.

    The benchmarking skill documents the overrides as optional, so `--arm base --arm
    off:HUGEPAGES=off` is an advertised invocation. It measured a service that had been
    up for an unknown time -- warm page cache, settled clocks, whatever a previous run
    left behind -- against one recreated seconds earlier, and neither artifact recorded
    the asymmetry. The bias runs warm-arm-faster, which is the direction that invents
    an improvement.
    """

    def test_a_bare_arm_is_recreated_like_its_partner(self, host, tmp_path):
        args = argv("--arm", "base", "--arm", "off:HUGEPAGES=off", out=tmp_path)
        assert run_cmd.cmd(args) == 0
        assert len(host.composed) == 2, "one arm measured a service nobody recreated"

    def test_a_single_unmodified_run_is_left_alone(self, host, tmp_path):
        """One arm has no partner, so there is no asymmetry to correct.

        Recreating anyway would spend a model reload -- minutes on the CPU service --
        to make a single run marginally colder, which is not what the finding is about.
        """
        assert run_cmd.cmd(argv(out=tmp_path)) == 0
        assert host.composed == []


class TestTheServedModelIsReconciled:
    """The registry's claim and the container's state, checked before measuring.

    A report carries conditions.model from bench/registry.py and conditions.server_argv
    from the container, and nothing reconciled them. The disagreement is reachable
    without any fault: the training commands swap models on these same services and
    skip their restore under --keep-serving, and an arm can override the model path.
    The result is a citable report describing a model that was not loaded.
    """

    def served(self, path):
        return lambda service, container_name=None: {
            "host": {"client_cpuset": "0-15", "service_cpuset": "0-15"},
            "env_knobs": {"MODEL_PATH": path},
            "server_argv": ["llama-server", "-m", path],
        }

    def test_a_service_running_another_model_is_refused(
        self, host, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "bench.conditions.collect", self.served("/models/something-else.gguf")
        )
        assert run_cmd.cmd(argv(out=tmp_path)) == 2
        assert host.engine.prompts == [], "a workload ran against the wrong model"
        assert list(tmp_path.glob("*/result.json")) == []

    def test_the_registered_path_is_accepted(self, host, tmp_path, monkeypatch):
        from bench.registry import resolve

        spec = resolve(MODEL, "llama-cpu")
        monkeypatch.setattr("bench.conditions.collect", self.served(spec.path))
        assert run_cmd.cmd(argv(out=tmp_path)) == 0

    def test_a_container_stating_no_model_is_not_contradicted(self, host, tmp_path):
        """docker inspect can fail, and unobserved is not the same as disagreeing.

        The fixture's block carries no model path at all, which is what a sandbox --
        or a container compose named differently -- produces.
        """
        assert run_cmd.cmd(argv(out=tmp_path)) == 0
