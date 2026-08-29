"""The run loop end to end, against a fake service.

No GPU, no container, no HTTP: a stub stream function stands in for the engine, which
lets the whole path -- cells, warm-up discard, reduction, envelope, file layout -- be
exercised in the sandbox. What cannot be proven here is any actual number.
"""

from __future__ import annotations

import json

import pytest

from bench import adapters, runner
from bench.client import StreamResult
from bench.registry import ModelSpec
from bench.report import read
from bench.workloads import load
from common.output import Printer
from common.reports import validate_report
from common.serving import ServingError

DENSE = ModelSpec(
    "d", "/m/x-Q4_K_M.gguf", ("llama-cpu",), "dense", 7.0, "Q4_K_M", 32768
)
MOE = ModelSpec(
    "m",
    "/m/y-Q4_K_M.gguf",
    ("llama-cpu",),
    "moe",
    30.5,
    "Q4_K_M",
    32768,
    params_active_b=3.3,
    n_experts=128,
    n_experts_active=8,
)


class FakeEngine:
    """A server that answers instantly, on a synthetic clock."""

    def __init__(self, ttft_s: float = 0.2, itl_s: float = 0.02, server_timings=True):
        self.ttft_s = ttft_s
        self.itl_s = itl_s
        self.server_timings = server_timings
        self.clock = 0.0
        self.prompts: list[str] = []

    def __call__(self, adapter, body, timeout):
        self.prompts.append(body["prompt"])
        n = body["max_tokens"]
        t_submit = self.clock
        arrivals = [t_submit + self.ttft_s + i * self.itl_s for i in range(n)]
        self.clock = arrivals[-1] + 1.0
        timings = None
        if self.server_timings:
            timings = adapters.ServerTimings(
                prompt_ms=self.ttft_s * 900, predicted_ms=self.itl_s * n * 1000
            )
        return StreamResult(
            t_submit=t_submit,
            arrivals=arrivals,
            usage=adapters.Usage(
                prompt_tokens=len(body["prompt"]) // 4, completion_tokens=n
            ),
            timings=timings,
            chunks_seen=n + 1,
        )


@pytest.fixture
def printer():
    return Printer(quiet=True)


def run(model=DENSE, engine=None, profile_name="quick"):
    engine = engine or FakeEngine()
    profile = load(profile_name, "llama-cpu")
    observations, errors = runner.execute(
        "llama-cpu", profile, model, Printer(quiet=True), stream_fn=engine
    )
    return profile, observations, errors, engine


class TestExecute:
    def test_warmups_are_run_and_discarded(self):
        profile, observations, _, engine = run()
        points = len(profile.points())
        # Every cell was sent, including the warm-ups...
        assert len(engine.prompts) == points * (profile.repeats + profile.warmup)
        # ...but only the measured ones are kept.
        assert len(observations) == points * profile.repeats
        assert all(o.cell.warmup is False for o in observations)

    def test_every_prompt_is_unique_so_a_prefix_cache_cannot_answer(self):
        _, _, _, engine = run()
        assert len(set(engine.prompts)) == len(engine.prompts)

    def test_the_nonce_leads_the_prompt_where_a_prefix_cache_keys(self):
        _, _, _, engine = run()
        assert all(p.startswith("[") for p in engine.prompts)

    def test_an_moe_draws_disjoint_prompt_regions(self):
        _, _, _, engine = run(model=MOE)
        decode = [p for p in engine.prompts if len(p) > 400]
        bodies = {p.split("] ", 1)[1][:64] for p in decode}
        # Three regions rotated across repeats, versus one for a dense model.
        assert len(bodies) == 3

    def test_a_dense_model_uses_one_region(self):
        _, _, _, engine = run(model=DENSE)
        bodies = {p.split("] ", 1)[1][:64] for p in engine.prompts}
        assert len(bodies) == 1

    def test_a_failing_cell_is_recorded_and_the_run_continues(self):
        engine = FakeEngine()
        calls = {"n": 0}
        original = engine.__call__

        def flaky(adapter, body, timeout):
            calls["n"] += 1
            if calls["n"] == 3:
                raise ServingError("connection reset")
            return original(adapter, body, timeout)

        profile = load("quick", "llama-cpu")
        observations, errors = runner.execute(
            "llama-cpu", profile, DENSE, Printer(quiet=True), stream_fn=flaky
        )
        assert len(errors) == 1
        assert "connection reset" in errors[0]["error"]
        assert observations

    def test_a_response_without_usage_is_refused_not_estimated(self):
        def no_usage(adapter, body, timeout):
            return StreamResult(
                0.0, [0.1, 0.2], usage=None, timings=None, chunks_seen=2
            )

        profile = load("quick", "llama-cpu")
        observations, errors = runner.execute(
            "llama-cpu", profile, DENSE, Printer(quiet=True), stream_fn=no_usage
        )
        assert not observations
        assert all("usage" in e["error"] for e in errors)


class TestReduce:
    def test_separates_the_phases(self):
        profile, observations, _, _ = run()
        phases = runner.reduce(observations, profile)
        assert set(phases) == {"prefill", "decode"}

    def test_the_prefill_arm_reports_no_decode_metrics(self):
        profile, observations, _, _ = run()
        prefill = runner.reduce(observations, profile)["prefill"]
        for point in prefill["points"]:
            assert point["max_tokens"] == 1
            assert point["metrics"]["decode_tok_s"] is None
            assert point["metrics"]["ttft_ms"]["median"] > 0

    def test_the_decode_arm_reports_a_rate_that_excludes_prefill(self):
        profile, observations, _, _ = run(engine=FakeEngine(ttft_s=1.0, itl_s=0.02))
        decode = runner.reduce(observations, profile)["decode"]["points"][0]
        # 50 tok/s from the 20ms inter-token gap, regardless of the 1s prefill; the
        # end-to-end figure is dragged down by that prefill and must differ.
        assert decode["metrics"]["decode_tok_s"]["median"] == pytest.approx(
            50.0, rel=1e-3
        )
        assert decode["metrics"]["e2e_tok_s"]["median"] < 50.0

    def test_the_prefill_slope_is_fitted_across_the_sweep(self):
        profile, observations, _, _ = run()
        prefill = runner.reduce(observations, profile)["prefill"]
        assert "slope" in prefill
        assert prefill["slope"]["points"] == len(profile.prefill_prompt_tokens)

    def test_timings_source_names_where_the_numbers_came_from(self):
        profile, observations, _, _ = run(engine=FakeEngine(server_timings=True))
        assert (
            runner.reduce(observations, profile)["prefill"]["timings_source"]
            == "server"
        )

        profile, observations, _, _ = run(engine=FakeEngine(server_timings=False))
        phases = runner.reduce(observations, profile)
        assert phases["prefill"]["timings_source"] == "client"
        # No substitute is invented where the engine reports nothing.
        for point in phases["prefill"]["points"]:
            assert point["metrics"]["prefill_tok_s_server"] is None

    def test_points_are_ordered_by_prompt_length(self):
        profile, observations, _, _ = run()
        points = runner.reduce(observations, profile)["prefill"]["points"]
        assert [p["prompt_tokens"] for p in points] == sorted(
            p["prompt_tokens"] for p in points
        )


class TestAssembleAndWrite:
    def test_the_document_validates_against_the_shared_envelope(self, tmp_path):
        profile, observations, errors, _ = run()
        document = runner.assemble(
            service="llama-cpu",
            profile=profile,
            model=MOE,
            arm="base",
            run_id="abc123",
            phases=runner.reduce(observations, profile),
            errors=errors,
            samples_path=None,
        )
        # docs/observability/report_schema.md requires this of every emitter.
        validate_report(document)
        assert document["report"] == "bench_result"
        assert document["schema_version"] == 1
        assert document["git_sha"]

    def test_conditions_carry_what_makes_a_number_citable(self, tmp_path):
        profile, observations, errors, _ = run()
        document = runner.assemble(
            service="llama-cpu",
            profile=profile,
            model=MOE,
            arm="base",
            run_id="abc123",
            phases=runner.reduce(observations, profile),
            errors=errors,
            samples_path=None,
        )
        conditions = document["conditions"]
        assert conditions["service"] == "llama-cpu"
        assert conditions["engine"] == "llama.cpp"
        assert conditions["prompt_source"] == "natural"
        assert conditions["model"]["architecture"] == "moe"
        assert conditions["model"]["params_active_b"] == 3.3
        assert conditions["model"]["quantization"] == "Q4_K_M"
        # Recording the server's own argv is what turns a silent misconfiguration --
        # an entrypoint execing a command it never built -- into a diffable field.
        assert "server_argv" in conditions
        assert "host" in conditions
        assert document["run"]["profile_sha256"] == profile.sha256
        assert document["run"]["warmup_discarded"] == profile.warmup

    def test_a_round_trip_through_the_reader_validates(self, tmp_path):
        from bench import report as report_mod

        profile, observations, errors, _ = run()
        document = runner.assemble(
            service="llama-cpu",
            profile=profile,
            model=DENSE,
            arm="base",
            run_id="abc123",
            phases=runner.reduce(observations, profile),
            errors=errors,
            samples_path=None,
        )
        path = report_mod.write(document, str(tmp_path))
        assert read(path)["run"]["arm"] == "base"

    def test_raw_samples_are_written_one_per_line(self, tmp_path):
        _, observations, _, _ = run()
        path = runner.write_samples(observations, str(tmp_path))
        lines = [json.loads(line) for line in open(path, encoding="utf-8")]
        assert len(lines) == len(observations)
        assert {"point", "repeat", "arrivals", "completion_tokens"} <= set(lines[0])
