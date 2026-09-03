"""Packaging hygiene: what is on disk must also be what ships.

These exist because of a real failure. An unanchored `registry/` line in
`.gitignore` matched `src/mlservice/registry/` as well as the intended top-level
output directory, so an entire package was silently left out of a commit. Every
local check still passed — the files were present in the working tree — and the
break only surfaced on a clean checkout.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
from pathlib import Path

import pytest

import mlservice

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(mlservice.__file__).resolve().parent


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


requires_git = pytest.mark.skipif(
    _git("rev-parse", "--git-dir") is None,
    reason="not a git checkout (an sdist, or git unavailable)",
)


@requires_git
def test_no_source_file_is_hidden_by_gitignore():
    """A .gitignore pattern must never swallow a file under src/ or tests/."""
    ignored = _git(
        "ls-files", "--others", "--ignored", "--exclude-standard", "--", "src", "tests", "scripts"
    )
    assert ignored is not None
    offenders = [
        line
        for line in ignored.splitlines()
        if line.strip() and "__pycache__" not in line and not line.endswith(".pyc")
    ]
    assert offenders == [], (
        "these source files are excluded by .gitignore and would not be committed: "
        f"{offenders}. Anchor the offending pattern to the repo root with a leading '/'."
    )


@requires_git
def test_every_python_module_under_src_is_tracked():
    tracked = _git("ls-files", "--", "src")
    assert tracked is not None
    tracked_set = {line.strip() for line in tracked.splitlines() if line.strip()}

    on_disk = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert on_disk - tracked_set == set(), "untracked source modules would not ship"


def test_every_subpackage_is_importable():
    """Catches a package that exists on disk but is missing from the distribution."""
    missing: list[str] = []
    for module in pkgutil.walk_packages(mlservice.__path__, prefix="mlservice."):
        try:
            importlib.import_module(module.name)
        except ImportError as exc:  # pragma: no cover - only on a broken install
            missing.append(f"{module.name}: {exc}")
    assert missing == [], f"installed package is incomplete: {missing}"


def test_the_documented_public_surface_imports():
    """The subpackages the README and docs tell people to use must be present."""
    for name in (
        "mlservice.config",
        "mlservice.artifacts",
        "mlservice.training",
        "mlservice.registry",
        "mlservice.serving",
        "mlservice.observability",
        "mlservice.monitoring",
        "mlservice.cli",
    ):
        importlib.import_module(name)
