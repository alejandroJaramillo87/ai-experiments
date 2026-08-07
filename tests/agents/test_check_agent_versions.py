"""Tests for scripts/agents/check-agent-versions.sh — npm pin drift check.

Contract: check-only, never edits the Dockerfile. Registry access is forced
to fail via a stubbed curl so the offline paths are testable; the live
lookup path is a host-side run.
"""

import json
import textwrap


def write_fixture(tmp_path, *args, plugin=None):
    """Write a Dockerfile of ARG pins plus the plugin package.json beside it.

    Returns the env dict that points the script at both. The plugin pin
    defaults to whatever OPENCODE_VERSION is set to, since matching it is what
    the coupling check wants; pass `plugin` to force a mismatch.
    """
    dockerfile = tmp_path / "Dockerfile.agents"
    lines = [f"ARG {a}" for a in args]
    dockerfile.write_text(textwrap.dedent("FROM scratch\n" + "\n".join(lines) + "\n"))

    if plugin is None:
        opencode = [a for a in args if a.startswith("OPENCODE_VERSION=")]
        plugin = opencode[0].split("=", 1)[1] if opencode else "0.0.0"
    package = tmp_path / "package.json"
    package.write_text(json.dumps({"devDependencies": {"@opencode-ai/plugin": plugin}}))

    return {"DOCKERFILE": str(dockerfile), "PLUGIN_PACKAGE_JSON": str(package)}


def test_offline_is_not_an_all_clear(run_script, fake_bin, tmp_path):
    # Every registry lookup fails: the summary must say unverified, never
    # print the [OK] all-clear.
    fake_bin.stub("curl", "exit 22")
    env = write_fixture(
        tmp_path,
        "OPENCODE_VERSION=1.0.0",
        "CLAUDE_CODE_VERSION=2.0.0",
        "PYRIGHT_VERSION=3.0.0",
    )
    result = run_script("agents/check-agent-versions.sh", env=env)
    assert result.returncode == 0
    assert "Pins unverified" in result.stdout
    assert "All pins current" not in result.stdout


def test_missing_pins_accumulate_and_fail(run_script, fake_bin, tmp_path):
    # All three missing ARG errors must print (no abort on the first) and
    # the script must exit nonzero.
    fake_bin.stub("curl", "exit 22")
    env = write_fixture(tmp_path)
    result = run_script("agents/check-agent-versions.sh", env=env)
    assert result.returncode == 1
    assert result.stdout.count("no ARG") == 3
    assert "Missing pins" in result.stdout


def test_plugin_pin_must_track_the_installed_opencode(run_script, fake_bin, tmp_path):
    # @opencode-ai/plugin publishes in lockstep with opencode-ai and the
    # plugins are typed against the runtime that loads them, so bumping
    # OPENCODE_VERSION alone must be reported.
    fake_bin.stub("curl", "exit 22")
    env = write_fixture(
        tmp_path,
        "OPENCODE_VERSION=1.2.0",
        "CLAUDE_CODE_VERSION=2.0.0",
        "PYRIGHT_VERSION=3.0.0",
        plugin="1.1.0",
    )
    result = run_script("agents/check-agent-versions.sh", env=env)
    assert "pinned 1.1.0, OPENCODE_VERSION is 1.2.0" in result.stdout


def test_a_wildcard_is_not_a_pin(run_script, fake_bin, tmp_path):
    # The operating rule is exact versions; a range would silently float.
    fake_bin.stub("curl", "exit 22")
    env = write_fixture(
        tmp_path,
        "OPENCODE_VERSION=1.2.0",
        "CLAUDE_CODE_VERSION=2.0.0",
        "PYRIGHT_VERSION=3.0.0",
        plugin="*",
    )
    result = run_script("agents/check-agent-versions.sh", env=env)
    assert result.returncode == 1
    assert "not an exact version" in result.stdout
