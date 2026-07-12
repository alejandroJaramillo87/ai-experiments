"""Tests for the Makefile worktree targets (create and clean).

First suite to drive Makefile targets directly. Each test runs `make -f
<repo>/Makefile` inside a throwaway git repository under tmp_path, so worktrees
land in tmp_path siblings and the real repository is never touched.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAKEFILE = REPO_ROOT / "Makefile"


@pytest.fixture
def worktree_repo(tmp_path) -> Path:
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


@pytest.fixture
def run_make(worktree_repo):
    def _run(*goals: str):
        return subprocess.run(
            ["make", "-C", str(worktree_repo), "-f", str(MAKEFILE), *goals],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "NO_COLOR": "1", "HOME": str(worktree_repo)},
            timeout=60,
        )

    return _run


def worktree_paths(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        line.split(" ", 1)[1]
        for line in out.splitlines()
        if line.startswith("worktree ")
    ]


def make_worktree(run_make, worktree_repo, name: str) -> Path:
    result = run_make(f"worktree-{name}")
    assert result.returncode == 0, result.stdout + result.stderr
    return worktree_repo.parent / f"ai-experiments-{name}"


class TestWorktreeCreate:
    def test_create(self, run_make, worktree_repo):
        result = run_make("worktree-foo")
        assert result.returncode == 0
        assert "[OK] worktree:" in result.stdout
        assert (worktree_repo.parent / "ai-experiments-foo").is_dir()
        branches = subprocess.run(
            ["git", "branch", "--list", "task/foo"],
            cwd=worktree_repo,
            capture_output=True,
            text=True,
        ).stdout
        assert "task/foo" in branches

    def test_create_idempotent(self, run_make, worktree_repo):
        make_worktree(run_make, worktree_repo, "foo")
        result = run_make("worktree-foo")
        assert result.returncode == 0
        assert "[SKIP] worktree already exists" in result.stdout

    def test_create_reuses_existing_task_branch(self, run_make, worktree_repo):
        # Simulate clean-without-branch-delete: branch exists, worktree does not
        subprocess.run(["git", "branch", "task/foo"], cwd=worktree_repo, check=True)
        result = run_make("worktree-foo")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "reusing existing branch task/foo" in result.stdout
        wt = worktree_repo.parent / "ai-experiments-foo"
        head = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=wt,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert head == "task/foo"

    def test_create_fails_on_stray_directory(self, run_make, worktree_repo):
        # e.g. docker auto-created the bind-mount path as a plain dir
        stray = worktree_repo.parent / "ai-experiments-foo"
        stray.mkdir()
        result = run_make("worktree-foo")
        assert result.returncode != 0
        assert "[ERROR]" in result.stdout
        assert "[OK]" not in result.stdout
        assert stray.is_dir()

    def test_create_no_false_skip_on_prefix(self, run_make, worktree_repo):
        # foo-bar existing must not make the unanchored path check skip foo
        make_worktree(run_make, worktree_repo, "foo-bar")
        result = run_make("worktree-foo")
        assert result.returncode == 0
        assert "[SKIP]" not in result.stdout
        assert (worktree_repo.parent / "ai-experiments-foo").is_dir()


class TestWorktreeClean:
    @pytest.mark.parametrize(
        "goals",
        [
            ("worktree-clean-foo",),
            ("worktree-clean", "foo"),
            ("worktree-clean", "NAME=foo"),
            ("worktree-clean", "task/foo"),
            ("worktree-clean", "../ai-experiments-foo"),
        ],
        ids=["suffix", "space", "name-var", "branch", "relative-path"],
    )
    def test_clean_forms(self, run_make, worktree_repo, goals):
        wt = make_worktree(run_make, worktree_repo, "foo")
        result = run_make(*goals)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[OK] removed worktree" in result.stdout
        assert not wt.exists()

    def test_clean_absolute_path(self, run_make, worktree_repo):
        wt = make_worktree(run_make, worktree_repo, "foo")
        result = run_make("worktree-clean", str(wt))
        assert result.returncode == 0
        assert not wt.exists()

    def test_bare_clean_errors_and_creates_nothing(self, run_make, worktree_repo):
        result = run_make("worktree-clean")
        assert result.returncode != 0
        assert "[ERROR] worktree-clean requires a name" in result.stdout
        assert not (worktree_repo.parent / "ai-experiments-clean").exists()

    def test_space_form_creates_no_worktree_for_argument(self, run_make, worktree_repo):
        make_worktree(run_make, worktree_repo, "foo")
        run_make("worktree-clean", "foo")
        paths = worktree_paths(worktree_repo)
        assert paths == [str(worktree_repo)]

    def test_dirty_worktree_fails_nonzero(self, run_make, worktree_repo):
        wt = make_worktree(run_make, worktree_repo, "foo")
        (wt / "dirty.txt").write_text("uncommitted\n")
        result = run_make("worktree-clean", "foo")
        assert result.returncode != 0
        assert "[ERROR] could not remove" in result.stdout
        assert "--force" in result.stdout
        assert wt.exists()

    def test_nonexistent_name_fails_nonzero(self, run_make):
        result = run_make("worktree-clean-nope")
        assert result.returncode != 0
        assert "[ERROR] could not remove" in result.stdout
