"""Structured logging.

JSON by default because the destination is a log aggregator, not a human. The
``console`` format exists for local development only. Request context is pulled
from contextvars rather than threaded through call signatures, so a log line
deep in the inference path still carries its request id.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from mlservice.config.observability import ObservabilityConfig
from mlservice.observability.context import current_context

#: LogRecord attributes that are not user-supplied extras.
#:
#: ``color_message`` is uvicorn's: it duplicates ``message`` with ANSI escapes
#: embedded, which is noise in a structured log and breaks naive log viewers.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with request context merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(current_context())

        extras = {key: value for key, value in record.__dict__.items() if key not in _RESERVED}
        if extras:
            payload.update(_jsonable(extras))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, separators=(",", ":"))


class ConsoleFormatter(logging.Formatter):
    """Human-readable local format; appends context as key=value pairs."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = current_context()
        if context:
            base += " " + " ".join(f"{key}={value}" for key, value in context.items())
        return base


def configure_logging(config: ObservabilityConfig | None = None) -> None:
    """Install the root handler. Idempotent: safe to call from tests and workers."""
    settings = config or ObservabilityConfig()
    formatter: logging.Formatter = (
        JsonFormatter() if settings.log_format == "json" else ConsoleFormatter()
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # uvicorn installs its own colourised handlers; route them through ours so
    # access logs are JSON too.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _jsonable(values: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, str | int | float | bool | type(None) | list | dict):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
