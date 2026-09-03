"""Drift monitoring extension points.

This package intentionally does **not** implement a drift detection product. It
defines the two seams a real one plugs into:

``DriftSink``
    Where scored predictions go. Swap the shipped JSONL sink for Kafka,
    BigQuery, S3, or a vendor SDK by writing one class.

``DriftDetector``
    How a stream of records is turned into a signal. Implement it in a batch
    job that reads what the sink wrote; the serving path stays synchronous and
    fast.

The serving path only ever touches ``DriftSink``, and only through
:class:`mlservice.monitoring.reporter.DriftReporter`, which is fail-open: a
monitoring outage must never take down inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from mlservice.monitoring.records import PredictionRecord

DriftSeverity = Literal["ok", "warn", "alert"]


class DriftSignal(BaseModel):
    """The output of a detector: one named measurement plus a verdict."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    threshold: float
    severity: DriftSeverity
    detail: dict[str, Any] = {}

    @property
    def is_drifting(self) -> bool:
        return self.severity != "ok"


class DriftSink(ABC):
    """Destination for prediction records emitted by the service."""

    @abstractmethod
    def emit(self, record: PredictionRecord) -> None:
        """Accept one record. Must be cheap; must not raise on transient failure."""

    def emit_batch(self, records: Sequence[PredictionRecord]) -> None:
        """Accept many records. Override when the backend supports bulk writes."""
        for record in records:
            self.emit(record)

    def flush(self) -> None:  # noqa: B027 - optional hook; most sinks buffer nothing
        """Push anything buffered. Called on graceful shutdown."""

    def close(self) -> None:  # noqa: B027 - optional hook; most sinks hold no resources
        """Release resources. Called on graceful shutdown, after flush()."""


class DriftDetector(ABC):
    """Turns observed records into signals, against a reference baseline.

    Implementations are expected to run offline against what a sink wrote, not
    inline in the request path.
    """

    @abstractmethod
    def evaluate(self, records: Sequence[PredictionRecord]) -> list[DriftSignal]:
        """Compare a window of records against the baseline."""
