"""Prometheus metrics.

Deliberately small: four service-health metrics and two model-behaviour metrics.
The model-behaviour ones matter most — a model degrades silently while latency
and error rate stay flat, so prediction score distribution and batch size are
the signals that catch a bad deploy.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST

#: Its own registry rather than the process-global default, so tests can build a
#: fresh set of metrics without duplicate-timeseries errors.
REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "mlservice_requests_total",
    "HTTP requests handled, by route and outcome.",
    labelnames=("method", "route", "status"),
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "mlservice_request_duration_seconds",
    "End-to-end HTTP request latency.",
    labelnames=("method", "route"),
    # Tuned for an in-process tabular model: most work lands under 50ms.
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

ERRORS = Counter(
    "mlservice_errors_total",
    "Handled failures, by class.",
    labelnames=("route", "kind"),
    registry=REGISTRY,
)

PREDICTIONS = Counter(
    "mlservice_predictions_total",
    "Individual rows scored (a batch of 50 increments this by 50).",
    labelnames=("model_name", "model_version"),
    registry=REGISTRY,
)

PREDICTION_SCORE = Histogram(
    "mlservice_prediction_score",
    "Distribution of predicted positive-class probabilities.",
    labelnames=("model_name", "model_version"),
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

BATCH_SIZE = Histogram(
    "mlservice_batch_size",
    "Rows per /predict call.",
    buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000),
    registry=REGISTRY,
)

MODEL_INFO = Gauge(
    "mlservice_model_loaded",
    "1 when a model is loaded and ready to serve.",
    labelnames=("model_name", "model_version", "stage"),
    registry=REGISTRY,
)


@contextmanager
def observe_latency(method: str, route: str) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        REQUEST_LATENCY.labels(method=method, route=route).observe(perf_counter() - started)


def record_request(method: str, route: str, status: int) -> None:
    REQUESTS.labels(method=method, route=route, status=str(status)).inc()


def record_error(route: str, kind: str) -> None:
    ERRORS.labels(route=route, kind=kind).inc()


def record_predictions(
    model_name: str,
    model_version: str,
    scores: list[float],
    *,
    observe_scores: bool = True,
) -> None:
    """Record a scored batch. ``observe_scores`` is the drift-distribution hook."""
    if not scores:
        return
    PREDICTIONS.labels(model_name=model_name, model_version=model_version).inc(len(scores))
    BATCH_SIZE.observe(len(scores))
    if observe_scores:
        histogram = PREDICTION_SCORE.labels(model_name=model_name, model_version=model_version)
        for score in scores:
            histogram.observe(score)


def set_model_loaded(model_name: str, model_version: str, stage: str) -> None:
    MODEL_INFO.labels(model_name=model_name, model_version=model_version, stage=stage).set(1)


def render() -> tuple[bytes, str]:
    """Scrape payload and its content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
