"""Registry abstraction.

The blueprint ships one implementation (a filesystem registry) but every
consumer — training, CLI, service — depends only on this interface. Swapping in
MLflow, SageMaker, or an S3-backed registry means writing one new subclass, not
touching the service.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlservice.artifacts.artifact import ModelArtifact

VERSION_PATTERN = re.compile(r"^v(\d+)$")

#: Conventional stages. Registries accept arbitrary strings; these are the ones
#: the shipped tooling and docs assume.
STAGE_PRODUCTION = "production"
STAGE_STAGING = "staging"
DEFAULT_STAGES = (STAGE_STAGING, STAGE_PRODUCTION)


class RegistryError(RuntimeError):
    """Base class for registry failures."""


class ModelNotFoundError(RegistryError):
    """No such model, version, or stage pointer."""


class VersionConflictError(RegistryError):
    """A version directory already exists, or a concurrent write won the race."""


class RegisteredVersion(BaseModel):
    """A registry entry: one immutable version of one model."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: str
    version: str
    uri: str = Field(
        description="Backend-specific locator; a directory path for the local backend."
    )
    created_at: datetime
    metrics: dict[str, float] = Field(default_factory=dict)
    git_commit: str | None = None
    passed_gates: bool = True
    stages: list[str] = Field(default_factory=list, description="Stages currently pointing here.")

    @property
    def version_number(self) -> int:
        match = VERSION_PATTERN.match(self.version)
        if not match:
            raise ValueError(f"Non-standard version string: {self.version}")
        return int(match.group(1))


class StageTransition(BaseModel):
    """One promotion or rollback, appended to an append-only audit log."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: str
    stage: str
    from_version: str | None
    to_version: str
    at: datetime
    reason: str = ""
    actor: str = "unknown"

    @property
    def is_rollback(self) -> bool:
        """True when the transition moved a stage to an older version."""
        if self.from_version is None:
            return False
        from_match = VERSION_PATTERN.match(self.from_version)
        to_match = VERSION_PATTERN.match(self.to_version)
        if not (from_match and to_match):
            return False
        return int(to_match.group(1)) < int(from_match.group(1))


class ModelRegistry(ABC):
    """Minimal registry contract.

    Implementations must guarantee that:

    * ``register`` allocates a new, previously unused version and never mutates
      an existing one — artifacts are immutable once registered.
    * stage pointers are separate from artifacts, so promoting is cheap and
      reversible.
    * every stage change is recorded in ``transitions``, which is what makes
      ``rollback`` possible without guessing.
    """

    @abstractmethod
    def register(
        self,
        artifact: ModelArtifact,
        *,
        version: str | None = None,
        allow_failed_gates: bool = False,
    ) -> RegisteredVersion:
        """Store an artifact under a fresh version and return its registry entry."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Names of all registered models."""

    @abstractmethod
    def list_versions(self, model_name: str) -> list[RegisteredVersion]:
        """All versions of a model, oldest first."""

    @abstractmethod
    def get_version(self, model_name: str, version: str) -> RegisteredVersion:
        """Registry entry for one exact version."""

    @abstractmethod
    def load(self, model_name: str, version: str) -> ModelArtifact:
        """Materialise an artifact for an exact version."""

    @abstractmethod
    def resolve_stage(self, model_name: str, stage: str) -> RegisteredVersion:
        """Registry entry a stage pointer currently refers to."""

    @abstractmethod
    def promote(
        self,
        model_name: str,
        version: str,
        stage: str,
        *,
        reason: str = "",
        actor: str = "unknown",
    ) -> StageTransition:
        """Point ``stage`` at ``version``."""

    @abstractmethod
    def transitions(self, model_name: str, stage: str | None = None) -> list[StageTransition]:
        """Stage-change audit log, oldest first."""

    @abstractmethod
    def delete_version(self, model_name: str, version: str) -> None:
        """Remove a version. Refuses while a stage still points at it."""

    # ---- behaviour shared by every backend --------------------------------

    def rollback(
        self, model_name: str, stage: str, *, reason: str = "", actor: str = "unknown"
    ) -> StageTransition:
        """Return ``stage`` to the version it pointed at before the last change.

        Implemented on the base class because it is pure history arithmetic: any
        backend that records transitions correctly gets a correct rollback.
        """
        history = self.transitions(model_name, stage)
        if not history:
            raise ModelNotFoundError(
                f"No transition history for {model_name}:{stage}; nothing to roll back to"
            )
        previous = history[-1].from_version
        if previous is None:
            raise ModelNotFoundError(
                f"{model_name}:{stage} has only ever pointed at one version; "
                "there is no earlier version to roll back to"
            )
        return self.promote(
            model_name,
            previous,
            stage,
            reason=reason or f"rollback from {history[-1].to_version}",
            actor=actor,
        )

    def load_stage(self, model_name: str, stage: str) -> ModelArtifact:
        """Convenience: resolve a stage and load the artifact behind it."""
        entry = self.resolve_stage(model_name, stage)
        return self.load(model_name, entry.version)

    def latest_version(self, model_name: str) -> RegisteredVersion:
        versions = self.list_versions(model_name)
        if not versions:
            raise ModelNotFoundError(f"Model '{model_name}' has no registered versions")
        return versions[-1]

    def describe(self, model_name: str) -> dict[str, Any]:
        """Everything a human needs about a model, in one call."""
        versions = self.list_versions(model_name)
        return {
            "model_name": model_name,
            "versions": [entry.model_dump(mode="json") for entry in versions],
            "transitions": [t.model_dump(mode="json") for t in self.transitions(model_name)],
        }
