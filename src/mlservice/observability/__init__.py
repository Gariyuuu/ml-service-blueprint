"""Observability: structured logs, Prometheus metrics, optional OTel tracing."""

from mlservice.observability.context import (
    current_context,
    get_request_id,
    new_request_id,
    set_model_version,
    set_request_id,
)
from mlservice.observability.logging import configure_logging, get_logger
from mlservice.observability.metrics import (
    record_error,
    record_predictions,
    record_request,
    render,
    set_model_loaded,
)
from mlservice.observability.tracing import configure_tracing, instrument_app, span

__all__ = [
    "configure_logging",
    "configure_tracing",
    "current_context",
    "get_logger",
    "get_request_id",
    "instrument_app",
    "new_request_id",
    "record_error",
    "record_predictions",
    "record_request",
    "render",
    "set_model_loaded",
    "set_model_version",
    "set_request_id",
    "span",
]
