"""Request-scoped context propagated into every log line and metric label."""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("mlservice_request_id", default=None)
_model_version: ContextVar[str | None] = ContextVar("mlservice_model_version", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


def set_model_version(version: str | None) -> None:
    _model_version.set(version)


def get_model_version() -> str | None:
    return _model_version.get()


def current_context() -> dict[str, str]:
    """Non-empty context fields, ready to merge into a log record."""
    return {
        key: value
        for key, value in (
            ("request_id", _request_id.get()),
            ("model_version", _model_version.get()),
        )
        if value is not None
    }
