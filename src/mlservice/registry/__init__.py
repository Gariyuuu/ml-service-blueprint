"""Model registry: version allocation, stage pointers, and rollback history."""

from mlservice.registry.base import (
    DEFAULT_STAGES,
    STAGE_PRODUCTION,
    STAGE_STAGING,
    ModelNotFoundError,
    ModelRegistry,
    RegisteredVersion,
    RegistryError,
    StageTransition,
    VersionConflictError,
)
from mlservice.registry.local import LocalFilesystemRegistry

__all__ = [
    "DEFAULT_STAGES",
    "STAGE_PRODUCTION",
    "STAGE_STAGING",
    "LocalFilesystemRegistry",
    "ModelNotFoundError",
    "ModelRegistry",
    "RegisteredVersion",
    "RegistryError",
    "StageTransition",
    "VersionConflictError",
]
