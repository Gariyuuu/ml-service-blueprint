"""Observability configuration: logging, metrics, tracing, and drift monitoring."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ObservabilityConfig(BaseSettings):
    """Overridable via ``MLSERVICE_OBS_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MLSERVICE_OBS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(
        default="json", description="'console' is human-readable; use 'json' everywhere else."
    )
    log_include_request_body: bool = Field(
        default=False,
        description="Log raw feature payloads. Off by default: inference inputs are often PII.",
    )

    metrics_enabled: bool = Field(default=True)
    metrics_path: str = Field(default="/metrics")
    prediction_histogram_enabled: bool = Field(
        default=True, description="Record a histogram of predicted scores for distribution drift."
    )

    tracing_enabled: bool = Field(
        default=False, description="Requires the 'otel' extra; a no-op when it is not installed."
    )
    otlp_endpoint: str | None = Field(
        default=None, description="e.g. http://otel-collector:4318/v1/traces"
    )
    trace_sample_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    service_namespace: str = Field(default="ml")

    drift_enabled: bool = Field(
        default=False, description="Emit prediction records to the configured drift sink."
    )
    drift_sink: Literal["null", "jsonl", "logging"] = Field(default="null")
    drift_sink_path: str = Field(
        default="var/drift/predictions.jsonl", description="Used when drift_sink is 'jsonl'."
    )
    drift_sample_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Fraction of predictions forwarded to the sink."
    )

    @model_validator(mode="after")
    def _tracing_needs_an_endpoint(self) -> ObservabilityConfig:
        if self.tracing_enabled and not self.otlp_endpoint:
            raise ValueError("tracing_enabled=True requires otlp_endpoint to be set")
        return self
