"""Tests for the permission policy in .claude/settings.json.

The allow list is expected to grow; the deny list is not. Nothing else in the
suite asserted on either, so an edit that dropped a deny rule while widening an
allow rule would have gone in green. These tests are the cheapest guard against
that, and they exist because the read allow-list was widened on 2026-08-08.
"""

import json

import pytest

SETTINGS = ".claude/settings.json"
LOCAL_SETTINGS = ".claude/settings.local.json"


@pytest.fixture
def settings(repo_root):
    return json.loads((repo_root / SETTINGS).read_text(encoding="utf-8"))


@pytest.fixture
def local_settings(repo_root):
    """The local file is git-ignored, so a fresh clone will not have one.

    Absent is the normal state, not a failure: it means nothing has been
    approved with "don't ask again" in this checkout yet.
    """
    path = repo_root / LOCAL_SETTINGS
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("rule", "why"),
    [
        ("Read(**/.env*)", "credentials and tokens"),
        ("Read(~/.ssh/**)", "private keys"),
        ("Read(~/.aws/**)", "cloud credentials"),
    ],
)
def test_key_material_denies_survive(settings, rule, why):
    """Deny precedes allow, so Read(**) is safe only while these are present."""
    assert rule in settings["permissions"]["deny"], why


@pytest.mark.parametrize("rule", ["Read(**)", "Grep(**)", "Glob(**)"])
def test_the_read_tools_are_allowed(settings, rule):
    assert rule in settings["permissions"]["allow"]


@pytest.mark.parametrize("tool", ["cat", "grep", "find"])
def test_no_blanket_bash_read_tool(settings, tool):
    """None of these is read-only: find -delete, grep -r > file, cat a > b.

    The dedicated Read/Grep/Glob tools cover the same ground and are matched by
    path rather than by command prefix, which is why the line sits here.
    """
    allow = settings["permissions"]["allow"]
    assert f"Bash({tool} *)" not in allow
    assert f"Bash({tool}*)" not in allow


def test_the_two_settings_files_do_not_overlap(settings, local_settings):
    """The effective policy is the union of both files.

    A rule in both is a rule whose owner is ambiguous: deleting it from the
    tracked file would silently leave it in force from the ignored one, in one
    checkout and not in another. Keep each rule in exactly one place.
    """
    tracked = set(settings["permissions"]["allow"])
    local = set(local_settings.get("permissions", {}).get("allow", []))
    assert not tracked & local
