"""Service configuration: runtime knobs for the inference API.

These are environment-first (12-factor) because they change per deployment,
unlike training config which is versioned alongside the experiment.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceConfig(BaseSettings):
    """Inference service settings, overridable via ``MLSERVICE_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MLSERVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        protected_namespaces=(),
    )

    app_name: str = Field(default="ml-service-blueprint")
    environment: Literal["local", "dev", "staging", "prod"] = Field(default="local")

    host: str = Field(default="0.0.0.0")  # noqa: S104 - containers must bind all interfaces
    port: int = Field(default=8000, ge=1, le=65535)

    registry_root: str = Field(
        default="registry", description="Filesystem registry the service loads models from."
    )
    model_name: str = Field(
        default="tabular-classifier", description="Registry key of the model to serve."
    )
    model_stage: str = Field(
        default="production",
        description="Stage pointer to resolve at startup. Ignored when model_version is set.",
    )
    model_version: str | None = Field(
        default=None, description="Pin an exact version, bypassing stage resolution."
    )

    max_batch_size: int = Field(
        default=1000, ge=1, description="Reject /predict payloads with more rows than this."
    )
    decision_threshold_override: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Override the threshold baked into the artifact. Leave unset in most cases.",
    )
    fail_fast_on_missing_model: bool = Field(
        default=True,
        description="Abort startup if no model resolves. Disable only for schema-less smoke tests.",
    )
    request_timeout_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _prod_should_not_override_threshold(self) -> ServiceConfig:
        if self.environment == "prod" and self.decision_threshold_override is not None:
            # Not fatal, but it means prod behaviour differs from what was validated at training.
            import warnings

            warnings.warn(
                "decision_threshold_override is set in prod; the served threshold no longer "
                "matches the one validated during training.",
                stacklevel=2,
            )
        return self
