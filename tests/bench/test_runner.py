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
DENSE_VLLM = ModelSpec(
    "dv", "/m/x-Q4_K_M.gguf", ("vllm-gpu",), "dense", 7.0, "Q4_K_M", 32768
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
        "llama-cpu",
        profile,
        model,
        Printer(quiet=True),
        run_key="base",
        stream_fn=engine,
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
            "llama-cpu",
            profile,
            DENSE,
            Printer(quiet=True),
            run_key="base",
            stream_fn=flaky,
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
            "llama-cpu",
            profile,
            DENSE,
            Printer(quiet=True),
            run_key="base",
            stream_fn=no_usage,
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
            arm_sequence="blocked",
            status=runner.COMPLETE,
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
            arm_sequence="blocked",
            status=runner.COMPLETE,
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
        assert document["run"]["profile"] == profile.name
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
            arm_sequence="blocked",
            status=runner.COMPLETE,
        )
        path = report_mod.write(document, str(tmp_path))
        assert read(path)["run"]["arm"] == "base"

    def test_raw_samples_are_written_one_per_line(self, tmp_path):
        _, observations, _, _ = run()
        path = runner.write_samples(observations, str(tmp_path))
        lines = [json.loads(line) for line in open(path, encoding="utf-8")]
        assert len(lines) == len(observations)
        assert {"point", "repeat", "arrivals", "completion_tokens"} <= set(lines[0])


class TestConcurrency:
    """A concurrency-N point must put N requests in flight, not run N serially.

    The absence of this test is what let the runner label a serial sweep
    `concurrency: 16`. A serial run reports queue-free latency under a queued label and
    one stream's throughput as a batched one -- wrong in the direction that flatters
    vLLM, which is the engine the sweep exists to measure.
    """

    def test_a_concurrent_cell_overlaps_its_requests_in_time(self):
        import threading
        import time as time_mod

        in_flight, peak = 0, 0
        lock = threading.Lock()

        def slow(adapter, body, timeout):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            time_mod.sleep(0.05)
            with lock:
                in_flight -= 1
            return FakeEngine()(adapter, body, timeout)

        profile = load("quick", "vllm-gpu")
        runner.execute(
            "vllm-gpu",
            profile,
            DENSE_VLLM,
            Printer(quiet=True),
            run_key="base",
            stream_fn=slow,
        )
        # quick.toml sweeps vllm-gpu at [1, 16]; serial execution would peak at 1.
        assert peak == 16, f"requests never overlapped (peak in flight: {peak})"

    def test_a_concurrent_cell_yields_one_sample_per_request(self):
        engine = FakeEngine()
        profile = load("quick", "vllm-gpu")
        observations, _ = runner.execute(
            "vllm-gpu",
            profile,
            DENSE_VLLM,
            Printer(quiet=True),
            run_key="base",
            stream_fn=engine,
        )
        by_point: dict[str, int] = {}
        for obs in observations:
            by_point[obs.cell.point.key] = by_point.get(obs.cell.point.key, 0) + 1
        c16 = next(k for k in by_point if k.endswith("/c16"))
        c1 = next(k for k in by_point if k.endswith("/c1") and k.startswith("decode"))
        # 16 in flight per repeat against 1, over the same repeat count.
        assert by_point[c16] == by_point[c1] * 16

    def test_concurrent_requests_do_not_share_a_prompt(self):
        # Identical prompts would let a prefix cache serve all but the first, and the
        # batch would measure the cache rather than the model.
        engine = FakeEngine()
        profile = load("quick", "vllm-gpu")
        runner.execute(
            "vllm-gpu",
            profile,
            DENSE_VLLM,
            Printer(quiet=True),
            run_key="base",
            stream_fn=engine,
        )
        assert len(set(engine.prompts)) == len(engine.prompts)

    def test_a_serial_point_still_runs_on_this_thread(self):
        engine = FakeEngine()
        profile = load("quick", "llama-cpu")
        observations, _ = runner.execute(
            "llama-cpu",
            profile,
            DENSE,
            Printer(quiet=True),
            run_key="base",
            stream_fn=engine,
        )
        assert len(observations) == len(profile.points()) * profile.repeats


class TestPromptIdentity:
    """Which prompts a pass sends, and what the nonce has to carry for that.

    The nonce is what defeats a prefix cache: vLLM caches on the leading tokens with
    no per-request override, so two passes that send the same bytes measure the cache
    on the second one. That makes prompt identity a property of the measurement, not
    of the string formatting, and it has to hold across every axis a pass varies on --
    across arms, across repeats, and whether or not the cell list was sliced.
    """

    def test_two_arms_in_one_process_share_no_prompt(self):
        # A noise pair runs both arms against one live service with no recreate
        # between them. A nonce built only from the seed and the position makes the
        # second arm a replay of the first, so the floor is a cold arm differenced
        # against a cached one -- too wide, which suppresses real findings.
        prompts = {}
        profile = load("quick", "vllm-gpu")
        for label in ("a", "a-prime"):
            engine = FakeEngine()
            runner.execute(
                "vllm-gpu",
                profile,
                DENSE_VLLM,
                Printer(quiet=True),
                run_key=label,
                stream_fn=engine,
            )
            prompts[label] = engine.prompts
        shared = set(prompts["a"]) & set(prompts["a-prime"])
        assert not shared, (
            f"{len(shared)} of {len(prompts['a'])} prompts are byte-identical between "
            "the two arms; the second arm measures the first arm's prefix cache"
        )

    def test_a_run_split_into_per_repeat_passes_repeats_no_prompt(self):
        # Interleaving arms means handing execute one repeat's cells at a time. A
        # nonce keyed on the enumerate position restarts with each pass, so every
        # repeat would send the first repeat's prompts.
        profile = load("quick", "llama-cpu")
        groups: dict[int, list] = {}
        for cell in profile.cells():
            groups.setdefault(cell.repeat, []).append(cell)

        sent: list[str] = []
        for group in groups.values():
            engine = FakeEngine()
            runner.execute(
                "llama-cpu",
                profile,
                DENSE,
                Printer(quiet=True),
                run_key="base",
                cells=group,
                stream_fn=engine,
            )
            sent.extend(engine.prompts)
        assert len(set(sent)) == len(sent), (
            f"{len(sent) - len(set(sent))} prompts repeated across per-repeat passes"
        )

    def test_a_cell_sends_the_same_prompt_sliced_or_whole(self):
        # The nonce is the cell's identity, not its offset in whatever list it
        # arrived in, so slicing the list cannot change what is sent.
        profile = load("quick", "llama-cpu")
        cells = profile.cells()

        whole = FakeEngine()
        runner.execute(
            "llama-cpu",
            profile,
            DENSE,
            Printer(quiet=True),
            run_key="base",
            cells=cells,
            stream_fn=whole,
        )
        tail = FakeEngine()
        runner.execute(
            "llama-cpu",
            profile,
            DENSE,
            Printer(quiet=True),
            run_key="base",
            cells=cells[-4:],
            stream_fn=tail,
        )
        assert tail.prompts == whole.prompts[-4:]

    def test_the_default_cell_list_is_the_whole_profile(self):
        profile = load("quick", "llama-cpu")
        engine = FakeEngine()
        runner.execute(
            "llama-cpu",
            profile,
            DENSE,
            Printer(quiet=True),
            run_key="base",
            stream_fn=engine,
        )
        assert len(engine.prompts) == len(profile.cells())


class LinearPrefillEngine:
    """A server whose TTFT is linear in the tokens *it* counted.

    `ratio` is server tokens per estimated token: the estimate that slices a prompt is
    four characters to the token, and no real BPE vocabulary agrees with it exactly. A
    slope fitted against the estimate is therefore a rate in a unit no engine reports,
    while `prefill_tok_s_server` in the same document is in real tokens.
    """

    def __init__(self, ms_per_token: float = 0.5, ratio: float = 2.0):
        self.ms_per_token = ms_per_token
        self.ratio = ratio
        self.clock = 0.0
        self.prompts: list[str] = []

    def __call__(self, adapter, body, timeout):
        self.prompts.append(body["prompt"])
        server_tokens = int(len(body["prompt"]) // 4 * self.ratio)
        ttft_s = server_tokens * self.ms_per_token / 1000.0
        n = body["max_tokens"]
        arrivals = [self.clock + ttft_s + i * 0.02 for i in range(n)]
        result = StreamResult(
            t_submit=self.clock,
            arrivals=arrivals,
            usage=adapters.Usage(prompt_tokens=server_tokens, completion_tokens=n),
            timings=None,
            chunks_seen=n + 1,
        )
        self.clock = arrivals[-1] + 1.0
        return result


class TestPrefillSlopeAxis:
    """What the prefill slope is a rate *in*.

    The slope and prefill_tok_s_server sit in one document and are both graded as
    throughput by bench compare, so they have to be in the same unit. The absolute
    figure is the one docs/optimizations/README.md publishes; an A/B delta cancels the
    rescale, a published rate does not.
    """

    def test_the_slope_is_a_rate_in_the_servers_tokens_not_the_estimate(self):
        # 0.5 ms per real prompt token is 2000 real tokens/s. The server counts two
        # tokens for every one the four-chars-per-token estimate predicts, so a fit
        # through the estimate reports 1000 -- half the truth, and in no unit at all.
        profile, observations, _, _ = run(engine=LinearPrefillEngine(0.5, 2.0))
        slope = runner.reduce(observations, profile)["prefill"]["slope"]
        assert slope["prefill_tok_s_slope"] == pytest.approx(2000.0, rel=1e-6)

    def test_the_slope_names_which_token_count_it_was_fitted_through(self):
        profile, observations, _, _ = run(engine=LinearPrefillEngine(0.5, 2.0))
        slope = runner.reduce(observations, profile)["prefill"]["slope"]
        assert slope["prompt_tokens_source"] == "observed"

    def test_the_estimate_is_the_fallback_when_the_server_counted_nothing(self):
        points = [
            {"prompt_tokens": 128, "metrics": {"ttft_ms": {"median": 64.0}}},
            {"prompt_tokens": 1024, "metrics": {"ttft_ms": {"median": 512.0}}},
        ]
        xs, source = runner.slope_axis(points)
        assert xs == [128.0, 1024.0]
        assert source == "estimated"

    def test_one_point_missing_its_observed_count_falls_the_whole_axis_back(self):
        # Mixing the two would put two units on one x-axis, which is worse than
        # either: the fit would be through a line that changes meaning halfway.
        points = [
            {
                "prompt_tokens": 128,
                "metrics": {
                    "ttft_ms": {"median": 64.0},
                    "prompt_tokens_observed": {"median": 140.0},
                },
            },
            {"prompt_tokens": 1024, "metrics": {"ttft_ms": {"median": 512.0}}},
        ]
        xs, source = runner.slope_axis(points)
        assert xs == [128.0, 1024.0]
        assert source == "estimated"
