"""Artifact contract: pipeline + schema + metrics + provenance + model card."""

from mlservice.artifacts.artifact import (
    ArtifactIntegrityError,
    ModelArtifact,
    PredictionResult,
)
from mlservice.artifacts.digest import sha256_file
from mlservice.artifacts.metadata import ArtifactMetadata, DatasetInfo
from mlservice.artifacts.provenance import CodeProvenance
from mlservice.artifacts.schema import (
    FeatureSchema,
    FeatureSpec,
    SchemaValidationError,
)

__all__ = [
    "ArtifactIntegrityError",
    "ArtifactMetadata",
    "CodeProvenance",
    "DatasetInfo",
    "FeatureSchema",
    "FeatureSpec",
    "ModelArtifact",
    "PredictionResult",
    "SchemaValidationError",
    "sha256_file",
]
