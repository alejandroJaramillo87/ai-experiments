"""Parser surface and stdout discipline."""

from __future__ import annotations

import json

import pytest

from bench.cli import build_parser
from bench.commands.run import parse_arm
from bench.workloads import ProfileError


class TestParser:
    @pytest.mark.parametrize(
        "argv",
        [
            ["run", "--service", "llama-cpu", "--model", "m"],
            ["run", "--service", "vllm-gpu", "--model", "m", "--profile", "publish"],
            [
                "run",
                "--service",
                "llama-gpu",
                "--model",
                "m",
                "--arm",
                "on:HUGEPAGES=on",
                "--arm",
                "off:HUGEPAGES=off",
            ],
            ["noise", "--service", "llama-cpu", "--model", "m"],
            ["compare", "a.json", "b.json"],
            ["compare", "a.json", "b.json", "--noise", "n.json", "--all"],
            ["models"],
            ["models", "--json"],
        ],
    )
    def test_documented_invocations_parse(self, argv):
        assert build_parser().parse_args(argv).command == argv[0]

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args([])
        assert exc.value.code == 2

    def test_an_unknown_service_is_rejected_by_the_parser(self):
        # The choices come from common/serving.py's registry, so a retired container
        # name fails at parse time rather than against a port that no longer exists.
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["run", "--service", "llama-hugepages", "--model", "m"]
            )

    def test_run_requires_a_model_so_no_result_lacks_its_architecture(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--service", "llama-cpu"])

    def test_viewer_commands_opt_out_of_the_run_prune(self):
        # A command that only displays must not delete the runs it was asked about.
        for argv in (["compare", "a.json", "b.json"], ["models"]):
            assert build_parser().parse_args(argv).obs_retention is False


class TestArmParsing:
    def test_a_bare_name_carries_no_overrides(self):
        assert parse_arm("base") == ("base", {})

    def test_overrides_are_parsed(self):
        assert parse_arm("off:HUGEPAGES=off") == ("off", {"HUGEPAGES": "off"})

    def test_several_overrides_in_one_arm(self):
        name, env = parse_arm("tuned:NO_REPACK=false,THREADS=12")
        assert name == "tuned"
        assert env == {"NO_REPACK": "false", "THREADS": "12"}

    def test_a_malformed_override_is_refused(self):
        with pytest.raises(ProfileError, match="KEY=VALUE"):
            parse_arm("bad:HUGEPAGES")

    def test_an_empty_name_defaults_to_base(self):
        assert parse_arm(":HUGEPAGES=off")[0] == "base"


class TestStdoutDiscipline:
    def test_models_json_puts_nothing_but_json_on_stdout(self, capsys):
        # .styles/PYTHON_OUTPUT_STYLE.md: stdout is reserved for machine output.
        from bench.commands.models import cmd

        args = build_parser().parse_args(["models", "--json"])
        assert cmd(args) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out)
        assert captured.err == ""

    def test_models_human_output_goes_to_stderr(self, capsys):
        from bench.commands.models import cmd

        assert cmd(build_parser().parse_args(["models"])) == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "[INFO]" in captured.err
