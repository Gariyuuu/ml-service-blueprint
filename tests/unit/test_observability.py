"""Structured logging, metrics, and tracing degradation."""

from __future__ import annotations

import json
import logging

from mlservice.config.observability import ObservabilityConfig
from mlservice.observability import metrics
from mlservice.observability.context import (
    current_context,
    new_request_id,
    set_model_version,
    set_request_id,
)
from mlservice.observability.logging import JsonFormatter, configure_logging
from mlservice.observability.tracing import configure_tracing, is_enabled, span


def _format(record_kwargs=None, message="hello"):
    record = logging.LogRecord("t", logging.INFO, __file__, 1, message, None, None)
    for key, value in (record_kwargs or {}).items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


def test_json_formatter_emits_one_parseable_object():
    payload = _format()
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert "timestamp" in payload


def test_json_formatter_merges_request_context():
    set_request_id("abc123")
    set_model_version("v9")
    try:
        payload = _format()
        assert payload["request_id"] == "abc123"
        assert payload["model_version"] == "v9"
    finally:
        set_request_id("")
        set_model_version(None)


def test_json_formatter_includes_extras():
    payload = _format({"duration_ms": 12.5, "route": "/predict"})
    assert payload["duration_ms"] == 12.5
    assert payload["route"] == "/predict"


def test_json_formatter_drops_uvicorns_ansi_duplicate():
    """uvicorn attaches color_message: the same text with ANSI escapes in it."""
    payload = _format({"color_message": "Started server process [\u001b[36m%d\u001b[0m]"})
    assert "color_message" not in payload


def test_json_formatter_stringifies_unserialisable_extras():
    payload = _format({"weird": object()})
    assert isinstance(payload["weird"], str)


def test_json_formatter_captures_exceptions():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
        payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_is_idempotent():
    configure_logging(ObservabilityConfig(log_format="json"))
    configure_logging(ObservabilityConfig(log_format="json"))
    assert len(logging.getLogger().handlers) == 1


def test_context_helpers_round_trip():
    request_id = new_request_id()
    set_request_id(request_id)
    assert current_context()["request_id"] == request_id


def test_metrics_render_is_prometheus_text():
    payload, content_type = metrics.render()
    assert b"mlservice_requests_total" in payload
    assert "text/plain" in content_type or "openmetrics" in content_type


def test_recording_predictions_moves_the_counter():
    def total() -> float:
        value = metrics.PREDICTIONS.labels(model_name="m", model_version="v1")._value.get()
        return float(value)

    before = total()
    metrics.record_predictions("m", "v1", [0.1, 0.9, 0.5])
    assert total() - before == 3.0


def test_recording_an_empty_batch_is_a_no_op():
    metrics.record_predictions("m", "v1", [])


def test_tracing_stays_off_when_disabled():
    configure_tracing(ObservabilityConfig(tracing_enabled=False), "svc")
    assert is_enabled() is False
    with span("noop", attribute=1):
        pass  # must not raise even with no tracer installed
