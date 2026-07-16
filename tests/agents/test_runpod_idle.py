"""Tests for scripts/agents/runpod-idle-check.sh — warm-worker spend guardrail.

Contract: non-blocking (always exits 0). RUNPOD_IDLE_FIXTURE feeds a canned API
response so the warning path is testable without RunPod credentials.
"""


def test_warm_fixture_warns(run_script, fixtures_dir):
    fixture = str(fixtures_dir / "runpod_warm_fixture.json")
    result = run_script(
        "agents/runpod-idle-check.sh", env={"RUNPOD_IDLE_FIXTURE": fixture}
    )
    assert result.returncode == 0
    assert "[WARN]" in result.stdout
    assert "llama-70b" in result.stdout


def test_idle_fixture_reports_no_spend(run_script, fixtures_dir):
    fixture = str(fixtures_dir / "runpod_idle_fixture.json")
    result = run_script(
        "agents/runpod-idle-check.sh", env={"RUNPOD_IDLE_FIXTURE": fixture}
    )
    assert result.returncode == 0
    assert "no warm workers, no active spend" in result.stdout


def test_report_mode_emits_spend_total(run_script, fixtures_dir):
    fixture = str(fixtures_dir / "runpod_warm_fixture.json")
    result = run_script(
        "agents/runpod-idle-check.sh",
        env={"RUNPOD_IDLE_FIXTURE": fixture, "RUNPOD_REPORT": "1"},
    )
    assert result.returncode == 0
    assert "runpod_spend_per_hr=0.4900" in result.stdout


def test_fractional_worker_count_warns(run_script, fixtures_dir):
    # The API can return fractional counts (workersStandby: 1.0); these must
    # normalize to integers instead of aborting into a false all-clear.
    fixture = str(fixtures_dir / "runpod_fractional_fixture.json")
    result = run_script(
        "agents/runpod-idle-check.sh", env={"RUNPOD_IDLE_FIXTURE": fixture}
    )
    assert result.returncode == 0
    assert "[WARN]" in result.stdout
    assert "no warm workers" not in result.stdout


def test_missing_fixture_skips(run_script):
    result = run_script(
        "agents/runpod-idle-check.sh", env={"RUNPOD_IDLE_FIXTURE": "/nonexistent.json"}
    )
    assert result.returncode == 0
    assert "[SKIP]" in result.stdout


def test_no_key_no_fixture_skips(run_script):
    # run_script merges os.environ, so blank the key explicitly; the script
    # treats an empty RUNPOD_API_KEY as unset.
    result = run_script(
        "agents/runpod-idle-check.sh",
        env={"RUNPOD_IDLE_FIXTURE": "", "RUNPOD_API_KEY": ""},
    )
    assert result.returncode == 0
    assert "[SKIP]" in result.stdout
    assert "RUNPOD_API_KEY not set" in result.stdout
