"""Artifact metadata: the JSON document that travels beside the pickled pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlservice.artifacts.provenance import CodeProvenance
from mlservice.artifacts.schema import FeatureSchema

METADATA_FORMAT_VERSION = 1


class DatasetInfo(BaseModel):
    """What the model was fitted on."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Path or URI of the training table.")
    content_sha256: str = Field(description="Digest of the training file, for reproducibility.")
    n_rows: int
    n_features: int
    n_train: int
    n_validation: int
    n_test: int
    class_balance: dict[str, float] = Field(default_factory=dict)


class ArtifactMetadata(BaseModel):
    """The full artifact contract, minus the serialized pipeline itself.

    Anything a consumer needs in order to decide whether to trust, promote, or
    roll back a model belongs here — not in a wiki page.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    format_version: int = METADATA_FORMAT_VERSION

    model_name: str
    version: str = Field(description="Registry version, e.g. 'v3'.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    feature_schema: FeatureSchema
    decision_threshold: float

    metrics: dict[str, float] = Field(default_factory=dict)
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    gate_failures: list[str] = Field(default_factory=list)

    dataset: DatasetInfo
    provenance: CodeProvenance

    training_config: dict[str, Any] = Field(default_factory=dict)
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)

    model_file: str = Field(default="model.joblib")
    model_file_sha256: str = Field(default="")
    model_card_file: str = Field(default="model_card.md")

    training_duration_seconds: float = 0.0

    @property
    def passed_gates(self) -> bool:
        return not self.gate_failures

    def summary(self) -> dict[str, Any]:
        """Compact form for /model-info and log lines."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "git_commit": self.provenance.git_commit,
            "decision_threshold": self.decision_threshold,
            "metrics": self.metrics,
            "n_features": len(self.feature_schema.features),
            "passed_gates": self.passed_gates,
        }
