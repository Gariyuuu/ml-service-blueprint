"""Provenance capture: it must be accurate, and it must never raise."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mlservice.artifacts.provenance import CodeProvenance


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "a.txt").write_text("hello")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-qm", "initial")
    return root


def test_capture_records_the_commit_and_branch(clean_repo: Path):
    provenance = CodeProvenance.capture(clean_repo)
    assert provenance.git_commit and len(provenance.git_commit) == 40
    assert provenance.git_branch


def test_a_clean_tree_reports_not_dirty(clean_repo: Path):
    """Empty `git status --porcelain` means clean, not unknown."""
    assert CodeProvenance.capture(clean_repo).git_dirty is False


def test_a_dirty_tree_reports_dirty(clean_repo: Path):
    (clean_repo / "a.txt").write_text("changed")
    assert CodeProvenance.capture(clean_repo).git_dirty is True


def test_a_clean_commit_is_reproducible(clean_repo: Path):
    assert CodeProvenance.capture(clean_repo).is_reproducible is True


def test_a_dirty_tree_is_not_reproducible(clean_repo: Path):
    (clean_repo / "a.txt").write_text("changed")
    assert CodeProvenance.capture(clean_repo).is_reproducible is False


def test_capture_outside_a_repo_does_not_raise(tmp_path: Path):
    """Provenance is best-effort: a tarball with no .git must still train."""
    provenance = CodeProvenance.capture(tmp_path)
    assert provenance.git_commit is None
    assert provenance.git_dirty is None
    assert provenance.is_reproducible is False
    assert provenance.python_version


def test_capture_always_records_the_environment(tmp_path: Path):
    provenance = CodeProvenance.capture(tmp_path)
    assert provenance.python_version and provenance.platform
    assert "scikit-learn" in provenance.packages


def test_provenance_round_trips_through_json(clean_repo: Path):
    provenance = CodeProvenance.capture(clean_repo)
    assert CodeProvenance.model_validate_json(provenance.model_dump_json()) == provenance
