"""Code and environment provenance captured at training time.

Answers the question you will actually ask in an incident: *which commit,
on which library versions, produced the model that is currently serving?*
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

TRACKED_PACKAGES = ("scikit-learn", "numpy", "pandas", "joblib", "mlservice")


class CodeProvenance(BaseModel):
    """Version metadata for the code and environment that produced an artifact."""

    model_config = ConfigDict(frozen=True)

    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = Field(
        default=None, description="True if the working tree had uncommitted changes."
    )
    python_version: str = Field(default_factory=platform.python_version)
    platform: str = Field(default_factory=platform.platform)
    packages: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def capture(cls, repo_root: Path | None = None) -> CodeProvenance:
        """Snapshot git and package state. Never raises: provenance is best-effort."""
        root = repo_root or Path.cwd()
        return cls(
            git_commit=_git(root, "rev-parse", "HEAD"),
            git_branch=_git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            git_dirty=_git_dirty(root),
            packages={
                name: version
                for name in TRACKED_PACKAGES
                if (version := _package_version(name)) is not None
            },
        )

    @property
    def is_reproducible(self) -> bool:
        """True when the artifact traces to a clean, identified commit."""
        return bool(self.git_commit) and self.git_dirty is False


def _git_output(root: Path, *args: str) -> str | None:
    """Raw stdout of a git command, or None if git could not answer.

    Empty output is preserved as "" rather than collapsed to None: for
    ``status --porcelain``, empty means *clean*, which is the opposite of
    unknown.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git(root: Path, *args: str) -> str | None:
    """Git output, with empty treated as absent — right for commit and branch."""
    return _git_output(root, *args) or None


def _git_dirty(root: Path) -> bool | None:
    status = _git_output(root, "status", "--porcelain")
    if status is None:
        return None
    return bool(status)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
