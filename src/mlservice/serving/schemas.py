"""Pydantic request/response models for the inference API.

Feature-level validation is deliberately *not* expressed as a Pydantic model.
The model's input columns are known only at load time, from the artifact's
frozen schema, so instances arrive as free-form mappings and are validated by
:meth:`mlservice.artifacts.schema.FeatureSchema.validate_frame` — the same code
path offline batch scoring uses.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Instance = dict[str, Any]


class PredictRequest(BaseModel):
    """One or more rows to score."""

    model_config = ConfigDict(protected_namespaces=())

    instances: list[Instance] = Field(
        min_length=1,
        description="Rows of raw features, keyed by the model's feature names.",
    )
    threshold: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Per-request decision threshold. Omit to use the artifact's own.",
    )
    return_features: bool = Field(
        default=False,
        description="Echo the validated feature values back. Off by default; inputs are often PII.",
    )


class Prediction(BaseModel):
    """The score and derived label for one row."""

    score: float = Field(description="Predicted probability of the positive class.")
    label: int = Field(description="1 when score >= threshold, else 0.")
    features: Instance | None = Field(
        default=None, description="Populated only when return_features is true."
    )


class PredictResponse(BaseModel):
    """A scored batch, tagged with exactly which model produced it."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    threshold: float
    predictions: list[Prediction]
    count: int
    request_id: str
    inference_ms: float = Field(description="Server-side scoring time, excluding transport.")


class HealthResponse(BaseModel):
    """Liveness. Answers 'is this process running', nothing more."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"] = "ok"
    service: str
    version: str
    environment: str
    uptime_seconds: float


class ReadyResponse(BaseModel):
    """Readiness. Answers 'can this process serve traffic right now'."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ready", "not_ready"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    detail: str | None = None


class FeatureInfo(BaseModel):
    """One input column, as advertised to clients."""

    name: str
    kind: str
    required: bool
    minimum: float | None = None
    maximum: float | None = None
    categories: list[str] | None = None


class ModelInfoResponse(BaseModel):
    """Everything a caller or an on-call engineer needs about the loaded model."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    stage: str | None
    loaded_at: str
    decision_threshold: float
    trained_at: str
    git_commit: str | None
    metrics: dict[str, float]
    dataset_sha256: str
    n_features: int
    features: list[FeatureInfo]
    example_instance: Instance
    model_card_available: bool


class ErrorDetail(BaseModel):
    """The single error shape every failing endpoint returns."""

    error: str = Field(description="Stable machine-readable error code.")
    message: str
    request_id: str | None = None
    details: list[str] = Field(default_factory=list)
