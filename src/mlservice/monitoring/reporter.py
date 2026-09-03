"""The single seam between the serving path and drift monitoring.

Two properties matter here and are enforced by this class rather than by
convention in the route handler:

1. **Fail-open.** A sink that raises, blocks, or fills a disk must degrade
   inference to "unmonitored", never to "down".
2. **Sampled.** At high request volume, emitting every row is the expensive part.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from typing import Any

from mlservice.config.observability import ObservabilityConfig
from mlservice.monitoring.base import DriftSink
from mlservice.monitoring.records import PredictionRecord
from mlservice.monitoring.sinks import JsonlSink, LoggingSink, NullSink
from mlservice.observability.context import get_request_id

logger = logging.getLogger(__name__)


def build_sink(config: ObservabilityConfig) -> DriftSink:
    """Construct the configured sink. Add cases here to plug in a real backend."""
    if not config.drift_enabled or config.drift_sink == "null":
        return NullSink()
    if config.drift_sink == "logging":
        return LoggingSink(include_features=config.log_include_request_body)
    if config.drift_sink == "jsonl":
        return JsonlSink(config.drift_sink_path, include_features=config.log_include_request_body)
    raise ValueError(f"Unknown drift sink: {config.drift_sink}")


class DriftReporter:
    """Wraps a sink with sampling and error isolation."""

    def __init__(self, sink: DriftSink, *, sample_ratio: float = 1.0, enabled: bool = True) -> None:
        self.sink = sink
        self.sample_ratio = sample_ratio
        self.enabled = enabled
        self._failures = 0

    @classmethod
    def from_config(cls, config: ObservabilityConfig) -> DriftReporter:
        return cls(
            build_sink(config),
            sample_ratio=config.drift_sample_ratio,
            enabled=config.drift_enabled,
        )

    def report(
        self,
        *,
        model_name: str,
        model_version: str,
        scores: Sequence[float],
        labels: Sequence[int],
        threshold: float,
        event_ids: Sequence[str],
        latency_ms: float | None = None,
        features: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        """Emit a scored batch. Returns how many records reached the sink."""
        if not self.enabled or not scores:
            return 0

        request_id = get_request_id()
        records: list[PredictionRecord] = []
        for index, (score, label) in enumerate(zip(scores, labels, strict=True)):
            if self.sample_ratio < 1.0 and random.random() > self.sample_ratio:  # noqa: S311
                continue
            records.append(
                PredictionRecord(
                    event_id=event_ids[index],
                    model_name=model_name,
                    model_version=model_version,
                    score=float(score),
                    label=int(label),
                    threshold=threshold,
                    request_id=request_id,
                    latency_ms=latency_ms,
                    features=features[index] if features is not None else None,
                )
            )

        if not records:
            return 0

        try:
            self.sink.emit_batch(records)
        except Exception:
            # Log once per 100 failures: a broken sink should not also flood logs.
            self._failures += 1
            if self._failures % 100 == 1:
                logger.exception(
                    "drift sink failed; predictions are unmonitored",
                    extra={"failure_count": self._failures},
                )
            return 0
        return len(records)

    def shutdown(self) -> None:
        try:
            self.sink.flush()
            self.sink.close()
        except Exception:
            logger.exception("drift sink failed to shut down cleanly")
