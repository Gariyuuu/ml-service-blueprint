"""Drift monitoring extension points, plus one reference sink and detector.

See ``docs/architecture.md`` for where a real drift system attaches.
"""

from mlservice.monitoring.base import DriftDetector, DriftSeverity, DriftSignal, DriftSink
from mlservice.monitoring.detectors import ScoreDistributionDrift, population_stability_index
from mlservice.monitoring.records import PredictionRecord
from mlservice.monitoring.reporter import DriftReporter, build_sink
from mlservice.monitoring.sinks import JsonlSink, LoggingSink, NullSink, read_jsonl

__all__ = [
    "DriftDetector",
    "DriftReporter",
    "DriftSeverity",
    "DriftSignal",
    "DriftSink",
    "JsonlSink",
    "LoggingSink",
    "NullSink",
    "PredictionRecord",
    "ScoreDistributionDrift",
    "build_sink",
    "population_stability_index",
    "read_jsonl",
]
