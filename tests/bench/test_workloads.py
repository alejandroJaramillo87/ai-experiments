"""Profile parsing and matrix expansion."""

from __future__ import annotations

import pytest

from bench.workloads import DECODE, PREFILL, ProfileError, load
from common.serving import SERVICES

SHIPPED = ("quick", "publish")


class TestShippedProfiles:
    @pytest.mark.parametrize("name", SHIPPED)
    @pytest.mark.parametrize("service", sorted(SERVICES))
    def test_every_service_resolves(self, name, service):
        # A benchable service with no concurrency entry would fail at run time on the
        # workstation, which is the worst place to discover a typo in a profile.
        profile = load(name, service)
        assert profile.concurrency
        assert profile.repeats >= 3  # the skill's floor: ">= 3 samples"
        assert profile.warmup >= 1  # "discard the first run"

    @pytest.mark.parametrize("name", SHIPPED)
    def test_only_vllm_gets_a_concurrency_sweep(self, name):
        # llama.cpp serves one request at a time and the harness serializes generation
        # behind a mutex (common/serving.py). A sweep against them would measure a queue.
        for service in ("llama-cpu", "llama-gpu", "harness"):
            assert load(name, service).concurrency == (1,)
        assert len(load(name, "vllm-gpu").concurrency) > 1

    @pytest.mark.parametrize("name", SHIPPED)
    def test_the_hash_identifies_the_profile(self, name):
        first = load(name, "llama-gpu")
        assert len(first.sha256) == 64
        assert first.sha256 == load(name, "llama-cpu").sha256

    def test_overrides_narrow_the_sweep_for_slow_services(self):
        # An 8192-token prefill on CPU is minutes, not seconds.
        assert 8192 in load("publish", "llama-gpu").prefill_prompt_tokens
        assert 8192 not in load("publish", "llama-cpu").prefill_prompt_tokens
        assert load("publish", "harness").decode_max_tokens == (64,)


class TestExpansion:
    def test_prefill_points_sweep_length_at_one_token(self):
        points = [
            p for p in load("publish", "llama-gpu").points() if p.phase == PREFILL
        ]
        assert [p.prompt_tokens for p in points] == [128, 512, 2048, 8192]
        assert {p.max_tokens for p in points} == {1}
        # Above concurrency 1, time to first token carries queue wait, and a slope
        # fitted through queue-contaminated points measures the scheduler.
        assert {p.concurrency for p in points} == {1}

    def test_decode_points_cross_generation_length_with_concurrency(self):
        points = [p for p in load("publish", "vllm-gpu").points() if p.phase == DECODE]
        assert {p.concurrency for p in points} == {1, 4, 16}
        assert {p.prompt_tokens for p in points} == {128}

    def test_warmups_come_first_and_are_marked(self):
        cells = load("quick", "llama-gpu").cells()
        warmups = [c for c in cells if c.warmup]
        assert warmups == cells[: len(warmups)]
        assert all(c.repeat == -1 for c in warmups)

    def test_repeats_are_round_robin_over_points_not_blocked(self):
        # Blocked ordering lets thermal or placement drift alias onto whichever point
        # happened to run last. Round-robin spreads it across all of them.
        profile = load("quick", "llama-gpu")
        measured = [c for c in profile.cells() if not c.warmup]
        n_points = len(profile.points())
        first_pass = [c.point.key for c in measured[:n_points]]
        second_pass = [c.point.key for c in measured[n_points : 2 * n_points]]
        assert len(set(first_pass)) == n_points
        assert first_pass == second_pass
        assert measured[0].repeat == 0
        assert measured[n_points].repeat == 1

    def test_cell_count_is_points_times_repeats_plus_warmups(self):
        profile = load("publish", "vllm-gpu")
        n_points = len(profile.points())
        assert len(profile.cells()) == n_points * (profile.repeats + profile.warmup)


class TestRefusals:
    def test_a_missing_profile_names_the_available_ones(self):
        with pytest.raises(ProfileError, match="available"):
            load("nonexistent", "llama-cpu")

    def test_a_service_with_no_concurrency_entry_is_refused(self, tmp_path):
        path = tmp_path / "p.toml"
        path.write_text(
            'name="p"\nrepeats=3\nwarmup=1\n'
            "[prefill]\nprompt_tokens=[128,512]\nmax_tokens=1\n"
            "[decode]\nprompt_tokens=128\nmax_tokens=[64]\n"
            "[concurrency]\nllama-cpu=[1]\n"
        )
        with pytest.raises(ProfileError, match="no concurrency"):
            load(str(path), "vllm-gpu")

    def test_a_single_prefill_point_is_refused(self, tmp_path):
        # One point cannot be fitted, and a single-point prefill rate charges the fixed
        # per-request overhead to the model.
        path = tmp_path / "p.toml"
        path.write_text(
            'name="p"\nrepeats=3\nwarmup=1\n'
            "[prefill]\nprompt_tokens=[512]\nmax_tokens=1\n"
            "[decode]\nprompt_tokens=128\nmax_tokens=[64]\n"
            "[concurrency]\nllama-cpu=[1]\n"
        )
        with pytest.raises(ProfileError, match="fewer than two distinct"):
            load(str(path), "llama-cpu")
