"""The record type that flows from the serving path into drift monitoring."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRecord(BaseModel):
    """One scored row, with enough context to be joined against outcomes later.

    ``features`` is optional and off by default in the shipped sinks: inference
    inputs are frequently personal data, and a drift pipeline that quietly
    duplicates them into a log file is a compliance problem, not a feature.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    event_id: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_name: str
    model_version: str
    score: float
    label: int
    threshold: float
    request_id: str | None = None
    latency_ms: float | None = None
    features: dict[str, Any] | None = None
