"""Tests for scripts/agents/guard.sh — the tool-call guardrail.

Contract: Claude-hook-shaped JSON on stdin; exit 2 with [BLOCKED] on stderr to
block the call, exit 0 to allow. Empty stdin allows (defensive).
"""

import json


def guard_input(tool_name: str, **tool_input) -> str:
    return json.dumps({"tool_name": tool_name, "tool_input": tool_input})


def test_bash_env_indirection_blocked(run_script, fixtures_dir):
    # The canonical fixture: .env reached through $(echo ...) indirection that
    # a pattern-based permission matcher misses.
    payload = (fixtures_dir / "guard_indirection_input.json").read_text()
    result = run_script("agents/guard.sh", stdin=payload)
    assert result.returncode == 2
    assert "[BLOCKED]" in result.stderr
    assert "environment file" in result.stderr


def test_read_env_file_blocked(run_script):
    result = run_script(
        "agents/guard.sh", stdin=guard_input("Read", file_path="/repo/.env")
    )
    assert result.returncode == 2
    assert "environment file" in result.stderr


def test_write_under_models_dir_blocked(run_script):
    result = run_script(
        "agents/guard.sh",
        stdin=guard_input("Write", file_path="/mnt/ai-data/models/x.gguf"),
    )
    assert result.returncode == 2
    assert "protected models dir" in result.stderr


def test_lowercase_write_under_models_dir_blocked(run_script):
    # OpenCode normalizes tool names to lowercase and pipes them to the same
    # guard (guard.sh:43,50 match "write" alongside "Write"). The models-dir
    # write policy must apply identically regardless of case.
    result = run_script(
        "agents/guard.sh",
        stdin=guard_input("write", file_path="/mnt/ai-data/models/x.gguf"),
    )
    assert result.returncode == 2
    assert "protected models dir" in result.stderr


def test_lowercase_bash_write_under_models_dir_blocked(run_script):
    # Lowercase "bash" (guard.sh:57) must trip the same models-dir write guard
    # as capitalized "Bash".
    result = run_script(
        "agents/guard.sh",
        stdin=guard_input("bash", command="touch /mnt/ai-data/models/x.gguf"),
    )
    assert result.returncode == 2
    assert "protected models dir" in result.stderr


def test_benign_bash_allowed(run_script):
    result = run_script("agents/guard.sh", stdin=guard_input("Bash", command="ls -la"))
    assert result.returncode == 0
    assert result.stderr == ""


def test_benign_read_allowed(run_script):
    result = run_script(
        "agents/guard.sh", stdin=guard_input("Read", file_path="README.md")
    )
    assert result.returncode == 0
    assert result.stderr == ""


def test_empty_stdin_allowed(run_script):
    result = run_script("agents/guard.sh", stdin="")
    assert result.returncode == 0


def test_malformed_payload_blocked(run_script):
    # Fail closed: an unparseable payload must exit 2, not surface jq's own
    # exit code (which the hook runner would treat as allow).
    result = run_script("agents/guard.sh", stdin='{"tool_name": "Bash", trunc')
    assert result.returncode == 2
    assert "[BLOCKED]" in result.stderr


def test_downloader_write_under_models_dir_blocked(run_script):
    for command in (
        "curl -o /mnt/ai-data/models/x.gguf https://example.com/x.gguf",
        "wget -P /mnt/ai-data/models https://example.com/x.gguf",
        "rsync x.gguf /mnt/ai-data/models/",
        "touch /mnt/ai-data/models/x.gguf",
    ):
        result = run_script(
            "agents/guard.sh", stdin=guard_input("Bash", command=command)
        )
        assert result.returncode == 2, command
        assert "protected models dir" in result.stderr, command


def test_models_dir_read_allowed(run_script):
    result = run_script(
        "agents/guard.sh",
        stdin=guard_input("Bash", command="ls -la /mnt/ai-data/models"),
    )
    assert result.returncode == 0
    assert result.stderr == ""
