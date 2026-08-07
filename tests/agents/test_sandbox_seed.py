"""Tests for the Claude Code config seeded by `make agent-sandbox`.

The sandbox home is ephemeral tmpfs, so the Makefile seeds ~/.claude.json and
~/.claude/settings.json on every launch. These tests run the real agent-sandbox
target in a throwaway git repo with a stubbed `docker` on PATH: the stub prints
the command the container would run instead of starting anything, and the test
executes the seed portion against a scratch HOME to validate the written JSON.

Do not test this target with `make -n`: the recipe contains `$(MAKE)`, which
GNU make executes even under -n, launching the real container.
"""

import json
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAKEFILE = REPO_ROOT / "Makefile"

DOCKER_STUB = """#!/bin/sh
# Print the final argument (the sh -lc command string) instead of running it.
for last; do :; done
printf 'STUB_CMD:%s\\n' "$last"
"""


@pytest.fixture
def sandbox_env(throwaway_repo, tmp_path):
    """Throwaway repo + stubbed docker; returns a runner for agent-sandbox."""
    repo = throwaway_repo

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(DOCKER_STUB)
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)

    home = tmp_path / "home"
    home.mkdir()

    # Pre-create the worktree: the recipe's inner $(MAKE) does not inherit -f,
    # so the creation branch cannot run against the throwaway repo.
    subprocess.run(
        ["make", "-C", str(repo), "-f", str(MAKEFILE), "worktree-seedtest"],
        capture_output=True,
        check=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(home)},
    )

    def run(*goals: str) -> tuple[subprocess.CompletedProcess, str]:
        result = subprocess.run(
            [
                "make",
                "-C",
                str(repo),
                "-f",
                str(MAKEFILE),
                "agent-sandbox",
                "AGENT=claude",
                "NAME=seedtest",
                "CLAUDE_CODE_OAUTH_TOKEN=dummy",
                *goals,
            ],
            capture_output=True,
            text=True,
            env={
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "NO_COLOR": "1",
                "HOME": str(home),
            },
            timeout=60,
        )
        cmd = ""
        for line in result.stdout.splitlines():
            if line.startswith("STUB_CMD:"):
                cmd = line[len("STUB_CMD:") :]
        return result, cmd

    def seeded_files(cmd: str) -> dict[str, dict]:
        """Execute the seed prefix of the container command against HOME."""
        seed, _, _ = cmd.partition("; claude")
        subprocess.run(
            ["sh", "-c", seed],
            check=True,
            env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        )
        return {
            ".claude.json": json.loads((home / ".claude.json").read_text()),
            "settings.json": json.loads(
                (home / ".claude" / "settings.json").read_text()
            ),
        }

    run.seeded_files = seeded_files
    return run


def test_seed_writes_valid_json_with_defaults(sandbox_env):
    result, cmd = sandbox_env()
    assert result.returncode == 0, result.stdout + result.stderr
    assert cmd, "docker stub was never invoked"
    files = sandbox_env.seeded_files(cmd)

    settings = files["settings.json"]
    assert settings["model"] == "claude-opus-5[1m]"
    assert settings["effortLevel"] == "low"
    assert settings["theme"] == "dark"

    state = files[".claude.json"]
    assert state["hasCompletedOnboarding"] is True
    # Without this flag the CLI re-pins Fable launch effort, overriding effortLevel
    assert state["unpinFable5LaunchEffort"] is True
    assert state["theme"] == "dark"
    options = state["additionalModelOptionsCache"]
    assert options[0]["value"] == "claude-fable-5[1m]"
    assert options[0]["label"] == "Fable"


def test_seed_honors_effort_and_theme_overrides(sandbox_env):
    result, cmd = sandbox_env("EFFORT=high", "CLAUDE_THEME=light")
    assert result.returncode == 0, result.stdout + result.stderr
    files = sandbox_env.seeded_files(cmd)
    assert files["settings.json"]["effortLevel"] == "high"
    assert files["settings.json"]["theme"] == "light"


def test_seed_honors_model_override(sandbox_env):
    result, cmd = sandbox_env("CLAUDE_MODEL=claude-opus-4-8")
    assert result.returncode == 0, result.stdout + result.stderr
    files = sandbox_env.seeded_files(cmd)
    assert files["settings.json"]["model"] == "claude-opus-4-8"


def test_model_flag_appended(sandbox_env):
    _, cmd = sandbox_env("MODEL=claude-fable-5[1m]")
    assert cmd.endswith('claude --model "claude-fable-5[1m]"')


def test_interactive_command_launches_claude(sandbox_env):
    _, cmd = sandbox_env()
    assert cmd.endswith("; claude")
