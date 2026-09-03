"""Filesystem-backed model registry.

Layout::

    <root>/
        <model_name>/
            versions/
                v1/                 artifact directory (model.joblib, metadata.json, ...)
                v2/
            stages.json             {"production": "v2", "staging": "v3"}
            transitions.jsonl       append-only stage-change log

Version allocation uses an exclusive directory create, so two trainers racing to
register never collide on the same version number. That is the only concurrency
guarantee this backend makes, and it is enough for a single build host; a shared
registry should use a real backend (see docs/architecture.md).
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from mlservice.artifacts.artifact import METADATA_FILE, ModelArtifact
from mlservice.artifacts.metadata import ArtifactMetadata
from mlservice.registry.base import (
    VERSION_PATTERN,
    ModelNotFoundError,
    ModelRegistry,
    RegisteredVersion,
    RegistryError,
    StageTransition,
    VersionConflictError,
)

STAGES_FILE = "stages.json"
TRANSITIONS_FILE = "transitions.jsonl"
VERSIONS_DIR = "versions"


class LocalFilesystemRegistry(ModelRegistry):
    """A registry that is just a directory. Commit it, rsync it, or mount it."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ---- paths ------------------------------------------------------------

    def _model_dir(self, model_name: str) -> Path:
        _validate_name(model_name)
        return self.root / model_name

    def _versions_dir(self, model_name: str) -> Path:
        return self._model_dir(model_name) / VERSIONS_DIR

    def version_uri(self, model_name: str, version: str) -> Path:
        _validate_version(version)
        return self._versions_dir(model_name) / version

    # ---- writes -----------------------------------------------------------

    def register(
        self,
        artifact: ModelArtifact,
        *,
        version: str | None = None,
        allow_failed_gates: bool = False,
    ) -> RegisteredVersion:
        model_name = artifact.metadata.model_name
        if not artifact.metadata.passed_gates and not allow_failed_gates:
            raise RegistryError(
                f"Refusing to register {model_name}: evaluation gates failed "
                f"({'; '.join(artifact.metadata.gate_failures)}). "
                "Pass allow_failed_gates=True to override deliberately."
            )

        versions_dir = self._versions_dir(model_name)
        versions_dir.mkdir(parents=True, exist_ok=True)

        target, assigned = self._claim_version(versions_dir, version)
        try:
            artifact.metadata = artifact.metadata.model_copy(update={"version": assigned})
            artifact.save(target)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        return self._entry(model_name, assigned, artifact.metadata)

    def _claim_version(self, versions_dir: Path, requested: str | None) -> tuple[Path, str]:
        """Reserve a version directory atomically; returns (path, version)."""
        if requested is not None:
            _validate_version(requested)
            target = versions_dir / requested
            try:
                target.mkdir(parents=False, exist_ok=False)
            except FileExistsError as exc:
                raise VersionConflictError(f"Version {requested} already exists") from exc
            return target, requested

        # Retry: another process may claim the same number between our scan and mkdir.
        for _ in range(100):
            next_number = _next_version_number(versions_dir)
            candidate = f"v{next_number}"
            target = versions_dir / candidate
            try:
                target.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            return target, candidate
        raise VersionConflictError("Could not allocate a version after 100 attempts")

    def promote(
        self,
        model_name: str,
        version: str,
        stage: str,
        *,
        reason: str = "",
        actor: str = "unknown",
    ) -> StageTransition:
        _validate_stage(stage)
        # Fails loudly if the version does not exist.
        self.get_version(model_name, version)

        stages = self._read_stages(model_name)
        previous = stages.get(stage)
        stages[stage] = version
        self._write_stages(model_name, stages)

        transition = StageTransition(
            model_name=model_name,
            stage=stage,
            from_version=previous,
            to_version=version,
            at=datetime.now(UTC),
            reason=reason,
            actor=actor or os.environ.get("USER", "unknown"),
        )
        self._append_transition(model_name, transition)
        return transition

    def delete_version(self, model_name: str, version: str) -> None:
        path = self.version_uri(model_name, version)
        if not path.is_dir():
            raise ModelNotFoundError(f"{model_name}:{version} is not registered")
        pinned = [stage for stage, at in self._read_stages(model_name).items() if at == version]
        if pinned:
            raise RegistryError(
                f"Refusing to delete {model_name}:{version}; still referenced by "
                f"stage(s) {', '.join(sorted(pinned))}"
            )
        shutil.rmtree(path)

    # ---- reads ------------------------------------------------------------

    def list_models(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir() and (entry / VERSIONS_DIR).is_dir()
        )

    def list_versions(self, model_name: str) -> list[RegisteredVersion]:
        versions_dir = self._versions_dir(model_name)
        if not versions_dir.is_dir():
            return []
        numbered: list[tuple[int, str]] = []
        for entry in versions_dir.iterdir():
            match = VERSION_PATTERN.match(entry.name)
            if entry.is_dir() and match and (entry / METADATA_FILE).is_file():
                numbered.append((int(match.group(1)), entry.name))
        return [self._entry(model_name, name) for _, name in sorted(numbered)]

    def get_version(self, model_name: str, version: str) -> RegisteredVersion:
        path = self.version_uri(model_name, version)
        if not (path / METADATA_FILE).is_file():
            raise ModelNotFoundError(f"{model_name}:{version} is not registered")
        return self._entry(model_name, version)

    def load(self, model_name: str, version: str) -> ModelArtifact:
        path = self.version_uri(model_name, version)
        if not path.is_dir():
            raise ModelNotFoundError(f"{model_name}:{version} is not registered")
        return ModelArtifact.load(path)

    def resolve_stage(self, model_name: str, stage: str) -> RegisteredVersion:
        version = self._read_stages(model_name).get(stage)
        if version is None:
            raise ModelNotFoundError(
                f"No version is assigned to {model_name}:{stage}. "
                f"Known stages: {sorted(self._read_stages(model_name)) or 'none'}"
            )
        return self.get_version(model_name, version)

    def stages(self, model_name: str) -> dict[str, str]:
        """Current stage -> version mapping."""
        return dict(self._read_stages(model_name))

    def transitions(self, model_name: str, stage: str | None = None) -> list[StageTransition]:
        path = self._model_dir(model_name) / TRANSITIONS_FILE
        if not path.is_file():
            return []
        records: list[StageTransition] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = StageTransition.model_validate_json(line)
            if stage is None or record.stage == stage:
                records.append(record)
        return records

    def read_metadata(self, model_name: str, version: str) -> ArtifactMetadata:
        """Metadata only — cheap enough to call for every version in a listing."""
        path = self.version_uri(model_name, version) / METADATA_FILE
        if not path.is_file():
            raise ModelNotFoundError(f"{model_name}:{version} is not registered")
        return ArtifactMetadata.model_validate_json(path.read_text(encoding="utf-8"))

    # ---- internals --------------------------------------------------------

    def _entry(
        self, model_name: str, version: str, metadata: ArtifactMetadata | None = None
    ) -> RegisteredVersion:
        meta = metadata if metadata is not None else self.read_metadata(model_name, version)
        pinned = [stage for stage, at in self._read_stages(model_name).items() if at == version]
        return RegisteredVersion(
            model_name=model_name,
            version=version,
            uri=str(self.version_uri(model_name, version)),
            created_at=meta.created_at,
            metrics=meta.metrics,
            git_commit=meta.provenance.git_commit,
            passed_gates=meta.passed_gates,
            stages=sorted(pinned),
        )

    def _read_stages(self, model_name: str) -> dict[str, str]:
        path = self._model_dir(model_name) / STAGES_FILE
        if not path.is_file():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise RegistryError(f"{path} is corrupt: expected a JSON object")
        return {str(key): str(value) for key, value in loaded.items()}

    def _write_stages(self, model_name: str, stages: dict[str, str]) -> None:
        path = self._model_dir(model_name) / STAGES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(stages, indent=2, sort_keys=True) + "\n")

    def _append_transition(self, model_name: str, transition: StageTransition) -> None:
        path = self._model_dir(model_name) / TRANSITIONS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(transition.model_dump_json() + "\n")


def _atomic_write(path: Path, content: str) -> None:
    """Write via a temp file + rename so a reader never sees a half-written file."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def _next_version_number(versions_dir: Path) -> int:
    numbers = [
        int(match.group(1))
        for entry in versions_dir.iterdir()
        if entry.is_dir() and (match := VERSION_PATTERN.match(entry.name))
    ]
    return max(numbers, default=0) + 1


def _validate_name(model_name: str) -> None:
    if not model_name or "/" in model_name or "\\" in model_name or model_name.startswith("."):
        raise RegistryError(f"Invalid model name: {model_name!r}")


def _validate_version(version: str) -> None:
    if not VERSION_PATTERN.match(version):
        raise RegistryError(f"Invalid version {version!r}; expected the form 'v1'")


def _validate_stage(stage: str) -> None:
    if not stage or "/" in stage or stage.strip() != stage:
        raise RegistryError(f"Invalid stage name: {stage!r}")
