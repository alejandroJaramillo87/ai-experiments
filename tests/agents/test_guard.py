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


def test_key_material_read_blocked(run_script):
    # settings.json denies Read(~/.ssh/**) and Read(~/.aws/**) by pattern. These
    # assert the hook denies the same paths independently, so the two layers the
    # documentation claims actually both exist.
    for path in (
        "/home/node/.ssh/id_ed25519",
        "/home/node/.aws/credentials",
        "/home/node/.aws/config",
        "/work/certs/server.pem",
        "/work/certs/client.p12",
        "/work/certs/tls.key",
    ):
        result = run_script(
            "agents/guard.sh", stdin=guard_input("Read", file_path=path)
        )
        assert result.returncode == 2, path
        assert "key material" in result.stderr, path


def test_key_material_indirection_blocked(run_script):
    # The reason the hook exists: the permission matcher cannot see through
    # $(...), so the same path reached indirectly must be caught on the command.
    for command in (
        "cat $(echo ~/.ssh/id_rsa)",
        "grep -r AKIA ~/.aws/credentials",
        "openssl x509 -in /work/certs/server.pem -noout",
    ):
        result = run_script(
            "agents/guard.sh", stdin=guard_input("Bash", command=command)
        )
        assert result.returncode == 2, command
        assert "key material" in result.stderr, command


def test_key_material_patterns_do_not_overreach(run_script):
    # A rule that blocks everything passes the two tests above and is useless.
    # Each case here names a construct the secret patterns nearly match:
    # a word ending in "key", a directory that is not .ssh, and ordinary work.
    for command in (
        "grep -rn api_keys docs/",
        "ls docs/sshd/",
        "make quality",
    ):
        result = run_script(
            "agents/guard.sh", stdin=guard_input("Bash", command=command)
        )
        assert result.returncode == 0, command
        assert result.stderr == "", command


def test_models_dir_read_allowed(run_script):
    result = run_script(
        "agents/guard.sh",
        stdin=guard_input("Bash", command="ls -la /mnt/ai-data/models"),
    )
    assert result.returncode == 0
    assert result.stderr == ""
