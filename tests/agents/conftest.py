"""Shared fixtures for the agents-script tests."""

import subprocess

import pytest


@pytest.fixture
def throwaway_repo(tmp_path):
    """Throwaway git repo named like the real one so path forms normalize."""
    repo = tmp_path / "ai-experiments"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo, check=True)
    (repo / "file.txt").write_text("content\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo
