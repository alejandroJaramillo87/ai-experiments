"""Profile parsing and matrix expansion."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import pytest

from bench.adapters import ENGINE_FOR_SERVICE
from bench.workloads import (
    BENCH_DIR,
    CHARS_PER_TOKEN,
    DECODE,
    MIN_PROMPT_TOKENS,
    ORDERS,
    PREFILL,
    ProfileError,
    build_prompt,
    corpus,
    load,
    profile_path,
    slices,
)

SHIPPED = ("quick", "publish")
# The benchmarked services, not every service in common/serving.py's registry: the
# harness is served there and measured by `make ladder` instead.
BENCHABLE = sorted(ENGINE_FOR_SERVICE)


class TestShippedProfiles:
    @pytest.mark.parametrize("name", SHIPPED)
    @pytest.mark.parametrize("service", BENCHABLE)
    def test_every_service_resolves(self, name, service):
        # A benchable service with no concurrency entry would fail at run time on the
        # workstation, which is the worst place to discover a typo in a profile.
        profile = load(name, service)
        assert profile.concurrency
        assert profile.repeats >= 3  # the skill's floor: ">= 3 samples"
        assert profile.warmup >= 1  # "discard the first run"

    @pytest.mark.parametrize("name", SHIPPED)
    def test_only_vllm_gets_a_concurrency_sweep(self, name):
        # llama.cpp serves one request at a time (common/serving.py), so a sweep
        # against it would measure a queue rather than a batching engine.
        for service in ("llama-cpu", "llama-gpu"):
            assert load(name, service).concurrency == (1,)
        assert len(load(name, "vllm-gpu").concurrency) > 1

    def test_overrides_narrow_the_sweep_for_slow_services(self):
        # An 8192-token prefill on CPU is minutes, not seconds.
        assert 8192 in load("publish", "llama-gpu").prefill_prompt_tokens
        assert 8192 not in load("publish", "llama-cpu").prefill_prompt_tokens


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

    def test_every_benchable_service_has_a_profile_entry(self):
        # A benchable service missing from a profile fails at run time on the
        # workstation, which is the worst place to find a typo.
        for name in SHIPPED:
            for service in BENCHABLE:
                assert load(name, service).concurrency

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


class TestPrompts:
    def test_the_corpus_is_the_committed_wikitext_split(self):
        # Read from the one copy the repository already has -- the same directory
        # docker-compose.inference.yaml mounts into llama-cpu for test_perplexity.sh --
        # rather than a second copy carried inside this package.
        from bench.workloads import CORPUS

        assert CORPUS.endswith(
            "scripts/benchmark/datasets/wikitext-2-raw/wiki.valid.raw"
        )
        assert len(corpus()) > 500_000

    def test_the_corpus_is_prose_not_section_headers(self):
        text = corpus()
        assert " = = " not in text
        # wikitext's tokenizer artefacts are folded back into ordinary punctuation.
        assert " @-@ " not in text and " @.@ " not in text

    def test_slices_are_disjoint(self):
        regions = slices(3)
        assert len(regions) == 3
        assert len({r[:200] for r in regions}) == 3

    def test_one_slice_is_the_whole_corpus(self):
        assert len(slices(1)[0]) == len(corpus())

    def test_zero_slices_is_refused(self):
        with pytest.raises(ValueError):
            slices(0)

    def test_a_prompt_is_about_the_requested_length(self):
        region = slices(1)[0]
        for target in (128, 1024, 8192):
            assert len(build_prompt(target, region, "n")) == target * 4

    def test_the_nonce_leads_where_a_prefix_cache_keys(self):
        # A unique suffix would not defeat a prefix cache; a unique prefix does.
        region = slices(1)[0]
        a = build_prompt(128, region, "r0")
        b = build_prompt(128, region, "r1")
        assert a != b
        assert a.startswith("[r0] ") and b.startswith("[r1] ")


TOP = 'name = "p"\nrepeats = 3\nwarmup = 1\norder = "interleaved"\nseed = 42\n'
PREFILL_TABLE = "[prefill]\nprompt_tokens = [128, 512]\nmax_tokens = 1\n"
DECODE_TABLE = "[decode]\nprompt_tokens = 128\nmax_tokens = [64]\n"
CONCURRENCY_TABLE = "[concurrency]\nllama-cpu = [1]\n"


def write_profile(
    tmp_path,
    *,
    top: str = TOP,
    prefill: str = PREFILL_TABLE,
    decode: str = DECODE_TABLE,
    concurrency: str = CONCURRENCY_TABLE,
    overrides: str = "",
) -> str:
    """A profile file on disk, valid unless a section is replaced with a broken one."""
    path = tmp_path / "p.toml"
    path.write_text("\n".join([top, prefill, decode, concurrency, overrides]))
    return str(path)


class TestWorkloadHash:
    """BQ-F34. The workload has an identity of its own, not just a declared name."""

    def test_one_file_resolved_for_two_services_hashes_differently(self):
        # publish.toml narrows the prefill sweep for llama-cpu. Same file, same declared
        # name, different workload -- which is exactly the case a name cannot express.
        cpu = load("publish", "llama-cpu")
        gpu = load("publish", "llama-gpu")
        assert cpu.name == gpu.name
        assert cpu.workload_hash != gpu.workload_hash

    def test_the_same_resolution_hashes_the_same(self):
        assert load("quick", "vllm-gpu").workload_hash == (
            load("quick", "vllm-gpu").workload_hash
        )

    def test_every_field_that_shapes_the_matrix_moves_the_hash(self):
        base = load("publish", "vllm-gpu")
        variants = {
            "repeats": 7,
            "warmup": 2,
            "order": "blocked",
            "seed": 7,
            "prefill_prompt_tokens": (128, 256),
            "prefill_max_tokens": 2,
            "decode_prompt_tokens": 256,
            "decode_max_tokens": (128,),
            "concurrency": (1, 8),
        }
        for field, value in variants.items():
            changed = dataclasses.replace(base, **{field: value})
            assert changed.workload_hash != base.workload_hash, field

    def test_the_declared_name_is_not_part_of_the_workload(self):
        # The name is what a file asserts about itself; the hash is what it resolves to.
        base = load("quick", "llama-gpu")
        assert dataclasses.replace(base, name="other").workload_hash == (
            base.workload_hash
        )

    def test_the_hash_is_stable_across_processes(self):
        # hash() is salted per interpreter, so a digest built on it would differ between
        # the run that writes a report and the run that reads it.
        root = os.path.dirname(BENCH_DIR)
        code = (
            "from bench.workloads import load; "
            "print(load('publish', 'llama-cpu').workload_hash)"
        )
        digests = set()
        for seed in ("0", "1", "random"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            digests.add(result.stdout.strip())
        assert digests == {load("publish", "llama-cpu").workload_hash}


class TestOrdering:
    """BQ-F35. The ordering selector names its values instead of failing open."""

    def test_an_unrecognised_order_is_refused_at_load(self, tmp_path):
        for value in ("Interleaved", "interleave", "round-robin", ""):
            path = write_profile(
                tmp_path, top=TOP.replace('"interleaved"', f'"{value}"')
            )
            with pytest.raises(ProfileError, match="order"):
                load(path, "llama-cpu")

    def test_both_documented_orders_load(self, tmp_path):
        for value in ORDERS:
            path = write_profile(
                tmp_path, top=TOP.replace('"interleaved"', f'"{value}"')
            )
            assert load(path, "llama-cpu").order == value

    def test_an_unknown_order_never_falls_through_to_blocked(self):
        # The selector used to test one string for equality and take blocked ordering
        # for everything else, so a typo silently bought the ordering the protocol
        # exists to forbid.
        profile = dataclasses.replace(load("quick", "llama-gpu"), order="Interleaved")
        with pytest.raises(ProfileError, match="order"):
            profile.cells()

    def test_blocked_order_runs_every_repeat_of_a_point_together(self, tmp_path):
        path = write_profile(tmp_path, top=TOP.replace('"interleaved"', '"blocked"'))
        profile = load(path, "llama-cpu")
        measured = [c for c in profile.cells() if not c.warmup]
        assert [c.repeat for c in measured[: profile.repeats]] == list(
            range(profile.repeats)
        )


class TestEmptyLists:
    """BQ-F36. An empty list is a value, so `is None` does not catch it."""

    def test_an_empty_concurrency_list_is_refused(self, tmp_path):
        path = write_profile(tmp_path, concurrency="[concurrency]\nllama-cpu = []\n")
        with pytest.raises(ProfileError, match="empty"):
            load(path, "llama-cpu")

    def test_an_empty_decode_generation_list_is_refused(self, tmp_path):
        path = write_profile(
            tmp_path, decode="[decode]\nprompt_tokens = 128\nmax_tokens = []\n"
        )
        with pytest.raises(ProfileError, match="empty"):
            load(path, "llama-cpu")

    def test_an_empty_prefill_sweep_is_refused(self, tmp_path):
        path = write_profile(
            tmp_path, prefill="[prefill]\nprompt_tokens = []\nmax_tokens = 1\n"
        )
        with pytest.raises(ProfileError):
            load(path, "llama-cpu")

    def test_an_override_cannot_empty_a_sweep(self, tmp_path):
        path = write_profile(
            tmp_path,
            overrides="[overrides.llama-cpu]\ndecode_max_tokens = []\n",
        )
        with pytest.raises(ProfileError, match="empty"):
            load(path, "llama-cpu")


class TestOverrideKeys:
    """BQ-F37. An override that reaches nothing is refused, not ignored."""

    def test_a_misspelled_override_key_is_refused(self, tmp_path):
        path = write_profile(
            tmp_path,
            overrides="[overrides.llama-cpu]\nprefil_prompt_tokens = [128, 256]\n",
        )
        with pytest.raises(ProfileError, match="prefil_prompt_tokens"):
            load(path, "llama-cpu")

    def test_an_override_for_an_unknown_service_is_refused(self, tmp_path):
        # Dead configuration: the profile narrows a sweep for a service that will never
        # read it, and the service it meant runs the wide one.
        path = write_profile(
            tmp_path,
            overrides="[overrides.llama-cp]\nprefill_prompt_tokens = [128, 256]\n",
        )
        with pytest.raises(ProfileError, match="llama-cp"):
            load(path, "llama-cpu")

    def test_an_override_is_refused_for_every_service_not_only_its_own(self, tmp_path):
        # The typo is in the file, so resolving it for another service must still fail.
        path = write_profile(
            tmp_path,
            concurrency="[concurrency]\nllama-cpu = [1]\nvllm-gpu = [1]\n",
            overrides="[overrides.llama-cpu]\ndecode_maxtokens = [64]\n",
        )
        with pytest.raises(ProfileError, match="decode_maxtokens"):
            load(path, "vllm-gpu")

    def test_a_correct_override_still_applies(self, tmp_path):
        path = write_profile(
            tmp_path,
            overrides="[overrides.llama-cpu]\nprefill_prompt_tokens = [64, 128]\n",
        )
        assert load(path, "llama-cpu").prefill_prompt_tokens == (64, 128)


class TestMalformedProfiles:
    """BQ-F39. Every malformation is a ProfileError, which the commands grade as 2."""

    def test_malformed_toml_is_refused(self, tmp_path):
        path = tmp_path / "p.toml"
        path.write_text('name = "p\nrepeats = 3\n')
        with pytest.raises(ProfileError, match="TOML"):
            load(str(path), "llama-cpu")

    def test_a_path_that_is_not_a_file_is_refused(self, tmp_path):
        with pytest.raises(ProfileError):
            load(str(tmp_path), "llama-cpu")

    @pytest.mark.parametrize(
        "missing", ["name", "repeats", "warmup", "prefill", "decode", "concurrency"]
    )
    def test_a_missing_top_level_key_names_itself(self, tmp_path, missing):
        parts = {
            "top": TOP,
            "prefill": PREFILL_TABLE,
            "decode": DECODE_TABLE,
            "concurrency": CONCURRENCY_TABLE,
        }
        if missing in ("name", "repeats", "warmup"):
            parts["top"] = "".join(
                line
                for line in TOP.splitlines(keepends=True)
                if not line.startswith(missing)
            )
        else:
            parts[missing] = ""
        with pytest.raises(ProfileError, match=missing):
            load(write_profile(tmp_path, **parts), "llama-cpu")

    @pytest.mark.parametrize("key", ["prompt_tokens", "max_tokens"])
    def test_a_missing_key_inside_a_phase_table_is_refused(self, tmp_path, key):
        table = "".join(
            line
            for line in PREFILL_TABLE.splitlines(keepends=True)
            if not line.startswith(key)
        )
        with pytest.raises(ProfileError, match=key):
            load(write_profile(tmp_path, prefill=table), "llama-cpu")

    def test_a_scalar_where_a_sweep_belongs_is_refused(self, tmp_path):
        path = write_profile(
            tmp_path, prefill="[prefill]\nprompt_tokens = 128\nmax_tokens = 1\n"
        )
        with pytest.raises(ProfileError, match="list"):
            load(path, "llama-cpu")

    def test_a_string_where_a_repeat_count_belongs_is_refused(self, tmp_path):
        path = write_profile(tmp_path, top=TOP.replace("repeats = 3", 'repeats = "3"'))
        with pytest.raises(ProfileError, match="repeats"):
            load(path, "llama-cpu")

    def test_a_concurrency_table_that_is_not_a_table_is_refused(self, tmp_path):
        path = write_profile(tmp_path, concurrency="concurrency = [1]\n")
        with pytest.raises(ProfileError, match="concurrency"):
            load(path, "llama-cpu")


class TestRanges:
    """BQ-F40. A value that would degrade the run is refused before the health gate."""

    def test_zero_repeats_is_refused(self, tmp_path):
        # Zero repeats loads, expands to warm-ups only, and `bench run` then reports
        # the empty result as every cell having failed.
        path = write_profile(tmp_path, top=TOP.replace("repeats = 3", "repeats = 0"))
        with pytest.raises(ProfileError, match="repeats"):
            load(path, "llama-cpu")

    def test_a_negative_warmup_is_refused(self, tmp_path):
        path = write_profile(tmp_path, top=TOP.replace("warmup = 1", "warmup = -1"))
        with pytest.raises(ProfileError, match="warmup"):
            load(path, "llama-cpu")

    def test_a_boolean_is_not_an_integer(self, tmp_path):
        path = write_profile(tmp_path, top=TOP.replace("repeats = 3", "repeats = true"))
        with pytest.raises(ProfileError, match="repeats"):
            load(path, "llama-cpu")

    def test_a_list_where_the_decode_prompt_length_belongs_is_refused(self, tmp_path):
        # It reached the report as a degenerate point identity: "decode/p[128]/g64/c1".
        path = write_profile(
            tmp_path, decode="[decode]\nprompt_tokens = [128]\nmax_tokens = [64]\n"
        )
        with pytest.raises(ProfileError, match="prompt_tokens"):
            load(path, "llama-cpu")

    @pytest.mark.parametrize("value", [0, -1])
    def test_a_non_positive_concurrency_is_refused(self, tmp_path, value):
        path = write_profile(
            tmp_path, concurrency=f"[concurrency]\nllama-cpu = [{value}]\n"
        )
        with pytest.raises(ProfileError, match="concurrency"):
            load(path, "llama-cpu")

    def test_a_zero_generation_length_is_refused(self, tmp_path):
        path = write_profile(
            tmp_path, decode="[decode]\nprompt_tokens = 128\nmax_tokens = [0]\n"
        )
        with pytest.raises(ProfileError, match="max_tokens"):
            load(path, "llama-cpu")

    def test_a_negative_prefill_length_is_refused(self, tmp_path):
        path = write_profile(
            tmp_path, prefill="[prefill]\nprompt_tokens = [-128, 512]\nmax_tokens = 1\n"
        )
        with pytest.raises(ProfileError, match="prompt_tokens"):
            load(path, "llama-cpu")

    def test_a_prompt_too_short_to_carry_the_nonce_is_refused(self, tmp_path):
        # Closes the only route to build_prompt's degenerate return (BQ-F42).
        path = write_profile(
            tmp_path, prefill="[prefill]\nprompt_tokens = [1, 512]\nmax_tokens = 1\n"
        )
        with pytest.raises(ProfileError, match="prompt_tokens"):
            load(path, "llama-cpu")
        assert all(
            n >= MIN_PROMPT_TOKENS
            for name in SHIPPED
            for service in BENCHABLE
            for n in load(name, service).prefill_prompt_tokens
        )


class TestDegeneratePrompts:
    """BQ-F42. A prompt that is only its own nonce is refused, not returned."""

    def test_an_empty_region_is_refused(self):
        with pytest.raises(ValueError, match="region"):
            build_prompt(128, "", "n")

    def test_a_target_the_nonce_fills_is_refused(self):
        # The length guard compared against a negative number, so both the slice and
        # the repeat produced nothing and the caller got a prompt of just the nonce.
        for target in (0, 1, -128):
            with pytest.raises(ValueError, match="nonce"):
                build_prompt(target, corpus(), "42-0-0")

    def test_a_short_region_is_repeated_to_the_full_length(self):
        prompt = build_prompt(128, "abc ", "n")
        assert len(prompt) == 128 * CHARS_PER_TOKEN
        assert prompt.startswith("[n] abc abc ")


class TestPublishProfileHeader:
    """BQ-F41. The header states the service count its own table has."""

    def test_the_header_counts_the_services_the_table_lists(self):
        with open(profile_path("publish"), encoding="utf-8") as handle:
            text = handle.read()
        assert "four services" not in text
        assert "all three benchmarked services" in text
        assert len(BENCHABLE) == 3


class TestThePromptFloorAgainstTheRealNonce:
    """The floor has to be sized against the nonce the runner actually builds.

    These two facts live in different modules and were changed by different hands:
    `MIN_PROMPT_TOKENS` is here, and the nonce format is in `bench/runner.py`. The floor
    was originally sized against a header of the shape `[seed-index-slot] `; the nonce
    later became the cell's full identity, which is nearly three times longer and grows
    with the arm name. Nothing failed, because both shipped profiles ask for 128 tokens
    and have eight times the margin -- but a hand-written profile at the floor would have
    got a prompt that was mostly nonce, and a long arm name would have raised out of
    `build_prompt` in the middle of a run.
    """

    # What bench/runner.py builds: seed, run key, point key, repeat, occurrence, slot.
    # The run key is the arm name plus the repeat ordinal under `--interleave`.
    WORST_CASE_NONCE = "42-repack-off-r4-decode/p128/g256/c16-4-2-15"

    def test_the_floor_leaves_a_prompt_that_is_mostly_prose(self):
        budget = MIN_PROMPT_TOKENS * CHARS_PER_TOKEN
        header = len(f"[{self.WORST_CASE_NONCE}] ")
        assert header < budget // 2, (
            f"at the floor the nonce header is {header} of {budget} characters, so a "
            "prompt at the minimum length measures the cache-defeating prefix rather "
            "than the corpus"
        )

    def test_a_prompt_at_the_floor_is_built_rather_than_refused(self):
        prompt = build_prompt(MIN_PROMPT_TOKENS, corpus()[:4096], self.WORST_CASE_NONCE)
        assert prompt.startswith(f"[{self.WORST_CASE_NONCE}] ")
        body = prompt[len(f"[{self.WORST_CASE_NONCE}] ") :]
        assert len(body) > len(self.WORST_CASE_NONCE)
