"""Shipped drift sinks. All three are references, not a monitoring product."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path

from mlservice.monitoring.base import DriftSink
from mlservice.monitoring.records import PredictionRecord

logger = logging.getLogger(__name__)


class NullSink(DriftSink):
    """Discards everything. The default, so monitoring is opt-in."""

    def emit(self, record: PredictionRecord) -> None:  # noqa: ARG002 - interface method
        return


class LoggingSink(DriftSink):
    """Emits records as structured log events.

    Useful when the log pipeline is already the path of least resistance to a
    warehouse. Feature values are dropped unless explicitly enabled.
    """

    def __init__(self, *, include_features: bool = False, level: int = logging.INFO) -> None:
        self.include_features = include_features
        self.level = level
        self._logger = logging.getLogger("mlservice.drift")

    def emit(self, record: PredictionRecord) -> None:
        payload = record.model_dump(mode="json")
        if not self.include_features:
            payload.pop("features", None)
        self._logger.log(self.level, "prediction", extra={"drift_record": payload})


class JsonlSink(DriftSink):
    """Appends newline-delimited JSON to a local file.

    Adequate for a single replica and a batch job that reads the file. It is not
    a durable pipeline: on multiple replicas each pod writes its own file, and
    nothing rotates them. Replace it before this matters.
    """

    def __init__(self, path: str | Path, *, include_features: bool = False) -> None:
        self.path = Path(path)
        self.include_features = include_features
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8")

    def emit(self, record: PredictionRecord) -> None:
        self.emit_batch([record])

    def emit_batch(self, records: Sequence[PredictionRecord]) -> None:
        if not records:
            return
        lines = []
        for record in records:
            payload = record.model_dump(mode="json")
            if not self.include_features:
                payload.pop("features", None)
            lines.append(json.dumps(payload, separators=(",", ":")))
        with self._lock:
            self._handle.write("\n".join(lines) + "\n")

    def flush(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()


def read_jsonl(path: str | Path) -> list[PredictionRecord]:
    """Load records a :class:`JsonlSink` wrote. Handy for offline detectors."""
    source = Path(path)
    if not source.is_file():
        return []
    return [
        PredictionRecord.model_validate_json(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
