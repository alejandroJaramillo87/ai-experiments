"""Every long-running verification script is an entry's, and every entry
naming one has it.

scripts/verification/ holds a runbook entry's command block as a script that
`make verify ENTRY=<name>` runs and logs. A script no entry cites is a
command block nobody records the output of; an entry naming a script that is
not in the tree hands the operator a target that fails. Both are held here,
lexically, the way check_make_targets.py holds `make` targets: the runbook
cites `make verify ENTRY=<name>`, and the name is the script's basename.
"""

from __future__ import annotations

import os
import re
import subprocess
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts", "verification")
RUNBOOKS_DIR = os.path.join(REPO_ROOT, "docs", "verification")
CITATION = re.compile(r"make verify ENTRY=([A-Za-z0-9_.-]+)")


def scripts() -> set[str]:
    return {
        name[: -len(".sh")] for name in os.listdir(SCRIPTS_DIR) if name.endswith(".sh")
    }


def citations() -> dict[str, set[str]]:
    """Entry name -> the runbook pages citing it, over every live page."""
    found: dict[str, set[str]] = {}
    for dirpath, _, names in os.walk(RUNBOOKS_DIR):
        if os.path.basename(dirpath) == "archive":
            continue
        for name in names:
            if not name.endswith(".md"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as f:
                for entry in CITATION.findall(f.read()):
                    found.setdefault(entry, set()).add(os.path.relpath(path, REPO_ROOT))
    return found


def test_every_script_is_cited_by_a_runbook():
    uncited = scripts() - set(citations())
    assert not uncited, f"scripts no runbook entry runs: {sorted(uncited)}"


def test_every_citation_names_a_script():
    missing = {
        entry: sorted(pages)
        for entry, pages in citations().items()
        if entry not in scripts()
    }
    assert not missing, f"runbooks naming a script that is not in the tree: {missing}"


def test_every_script_is_executable_bash():
    for name in sorted(os.listdir(SCRIPTS_DIR)):
        if not name.endswith(".sh"):
            continue
        path = os.path.join(SCRIPTS_DIR, name)
        assert os.access(path, os.X_OK), f"{name} is not executable"
        with open(path, encoding="utf-8") as f:
            assert f.readline().startswith("#!/usr/bin/env bash"), name


def test_verify_logs_the_script_and_returns_its_exit(tmp_path):
    """The target's pipe into tee must carry the script's exit, under the
    /bin/sh make runs recipes with. That shell is dash on the workstation
    and here, and dash has no pipefail: FT.organisms's first attempt
    (2026-09-26) died at `set: Illegal option -o pipefail` before the script
    ran (KB-97). A stand-in script that prints and exits 3 proves the exit
    reaches make, the output reaches the log, and ARGS reach the script.

    The stand-in also runs a sub-make, because the real scripts are made of
    them: make hands command-line variable definitions to every sub-make
    through MAKEFLAGS, so without the target dropping it, ARGS would reach
    each stage's target as its own ARGS. FT.organisms's third attempt
    (2026-09-26) saw organism-generate run with its arguments doubled
    (KB-98); the sub-make here must see an empty ARGS."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "probe.mk").write_text(
        "probe:\n\t@printf 'sub-make ARGS=[%s]\\n' \"$(ARGS)\"\n"
    )
    (scripts / "stand-in.sh").write_text(
        '#!/usr/bin/env bash\nprintf "[INFO] args: %s\\n" "$*"\n'
        'make -s -f "$(dirname "$0")/probe.mk" probe\nexit 3\n'
    )
    env = {k: v for k, v in os.environ.items() if k != "AI_SANDBOX"}
    proc = subprocess.run(
        [
            "make",
            "verify",
            "ENTRY=stand-in",
            f"VERIFY_SCRIPTS={scripts}",
            f"VERIFY_LOG_DIR={tmp_path / 'logs'}",
            "ARGS=--one two",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    # make reports a failed recipe as its own exit 2 and names the recipe's
    # status on stderr; the target prints the script's exit on stdout.
    assert proc.returncode != 0, proc.stdout
    assert "Error 3" in proc.stderr, proc.stderr
    assert "[INFO] args: --one two" in proc.stdout
    assert "exit=3" in proc.stdout
    logs = list((tmp_path / "logs").glob("stand-in_*.log"))
    assert len(logs) == 1, logs
    text = logs[0].read_text()
    assert "[INFO] args: --one two" in text
    assert "sub-make ARGS=[]" in text, text


def test_verify_detached_writes_the_log_and_its_exit(tmp_path):
    """DETACH=1 returns at once, and the log ends with the script's exit
    once it finishes, so an operator can close the terminal on a run of
    hours and read the same paste later."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "stand-in.sh").write_text(
        '#!/usr/bin/env bash\nprintf "[INFO] args: %s\\n" "$*"\nexit 3\n'
    )
    env = {k: v for k, v in os.environ.items() if k != "AI_SANDBOX"}
    proc = subprocess.run(
        [
            "make",
            "verify",
            "ENTRY=stand-in",
            "DETACH=1",
            f"VERIFY_SCRIPTS={scripts}",
            f"VERIFY_LOG_DIR={tmp_path / 'logs'}",
            "ARGS=--one two",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[INFO] Detached: pid " in proc.stdout
    logs = list((tmp_path / "logs").glob("stand-in_*.log"))
    assert len(logs) == 1, logs
    for _ in range(100):
        text = logs[0].read_text()
        if text.endswith("exit=3\n"):
            break
        time.sleep(0.1)
    assert "[INFO] args: --one two\n" in text, text
    assert text.endswith("exit=3\n"), text


def test_verify_refuses_in_the_sandbox():
    """The target names the workstation as the remedy when AI_SANDBOX is set,
    before touching a log directory that does not exist here."""
    proc = subprocess.run(
        ["make", "verify", "ENTRY=ft-organisms"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "AI_SANDBOX": "1"},
    )
    assert proc.returncode != 0
    assert "[ERROR] verify: this target requires the workstation" in proc.stdout
