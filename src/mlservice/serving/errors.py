"""Exception handlers: one error shape for every failure mode."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from mlservice.artifacts.artifact import ArtifactIntegrityError
from mlservice.artifacts.schema import SchemaValidationError
from mlservice.observability.context import get_request_id
from mlservice.registry.base import ModelNotFoundError
from mlservice.serving.model_holder import ModelNotLoadedError
from mlservice.serving.schemas import ErrorDetail

logger = logging.getLogger(__name__)


def _error(
    code: str, message: str, http_status: int, details: list[str] | None = None
) -> JSONResponse:
    body = ErrorDetail(
        error=code, message=message, request_id=get_request_id(), details=details or []
    )
    return JSONResponse(status_code=http_status, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers. Ordering does not matter; FastAPI dispatches by type."""

    @app.exception_handler(SchemaValidationError)
    async def _schema_error(request: Request, exc: SchemaValidationError) -> JSONResponse:
        # 422, not 400: the request was well-formed JSON that failed the model's
        # feature contract. The caller needs the full list to fix it in one pass.
        logger.warning("schema validation failed", extra={"errors": exc.errors})
        return _error(
            "feature_schema_violation",
            "Request features do not satisfy the model's schema.",
            422,
            exc.errors,
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
        return _error(
            "invalid_request",
            "Request body failed validation.",
            422,
            details,
        )

    @app.exception_handler(ModelNotLoadedError)
    async def _not_loaded(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
        # 503 with no model is a readiness problem, not a client problem.
        return _error(
            "model_not_loaded",
            "No model is currently loaded; the service is not ready.",
            503,
            [str(exc)],
        )

    @app.exception_handler(ModelNotFoundError)
    async def _not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return _error("model_not_found", str(exc), 404)

    @app.exception_handler(ArtifactIntegrityError)
    async def _integrity(request: Request, exc: ArtifactIntegrityError) -> JSONResponse:
        logger.error("artifact integrity failure: %s", exc)
        return _error(
            "artifact_integrity_error",
            "The model artifact on disk does not match its recorded digest.",
            500,
            [str(exc)],
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error("http_error", str(exc.detail), exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception")
        # The message is deliberately generic: internal exception text can leak
        # file paths, queries, and feature values. The request id is the join key.
        return _error(
            "internal_error",
            "An unexpected error occurred.",
            500,
        )
