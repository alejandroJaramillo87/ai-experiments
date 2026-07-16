"""Tests for scripts/agents/check-agent-versions.sh — npm pin drift check.

Contract: check-only, never edits the Dockerfile. Registry access is forced
to fail via a stubbed curl so the offline paths are testable; the live
lookup path is a host-side run.
"""

import textwrap


def write_dockerfile(tmp_path, *args):
    path = tmp_path / "Dockerfile.agents"
    lines = [f"ARG {a}" for a in args]
    path.write_text(textwrap.dedent("FROM scratch\n" + "\n".join(lines) + "\n"))
    return str(path)


def test_offline_is_not_an_all_clear(run_script, fake_bin, tmp_path):
    # Every registry lookup fails: the summary must say unverified, never
    # print the [OK] all-clear.
    fake_bin.stub("curl", "exit 22")
    dockerfile = write_dockerfile(
        tmp_path, "OPENCODE_VERSION=1.0.0", "CLAUDE_CODE_VERSION=2.0.0",
        "PYRIGHT_VERSION=3.0.0")
    result = run_script(
        "agents/check-agent-versions.sh", env={"DOCKERFILE": dockerfile})
    assert result.returncode == 0
    assert "Pins unverified" in result.stdout
    assert "All pins current" not in result.stdout


def test_missing_pins_accumulate_and_fail(run_script, fake_bin, tmp_path):
    # All three missing ARG errors must print (no abort on the first) and
    # the script must exit nonzero.
    fake_bin.stub("curl", "exit 22")
    dockerfile = write_dockerfile(tmp_path)
    result = run_script(
        "agents/check-agent-versions.sh", env={"DOCKERFILE": dockerfile})
    assert result.returncode == 1
    assert result.stdout.count("no ARG") == 3
    assert "Missing pins" in result.stdout
