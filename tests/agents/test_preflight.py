"""Tests for scripts/agents/preflight-scan.sh — agent supply-chain scan.

Contract: extern mode (target is not this repo's root) exits 1 and flags every
vector; self mode (target is the repo root) is advisory and exits 0; a missing
target dir exits 2.
"""


def test_fixture_dir_flags_both_vectors(run_script, fixtures_dir):
    result = run_script(
        "agents/preflight-scan.sh", str(fixtures_dir / "preflight_fixture")
    )
    assert result.returncode == 1
    assert "evil.ts" in result.stdout
    assert "defines hooks" in result.stdout


def test_repo_root_is_advisory(run_script, repo_root):
    result = run_script("agents/preflight-scan.sh", str(repo_root))
    assert result.returncode == 0
    assert "trusted home repo" in result.stdout


def test_missing_target_dir(run_script):
    result = run_script("agents/preflight-scan.sh", "/nonexistent-preflight-target")
    assert result.returncode == 2
    assert "not found" in result.stdout
