"""FastAPI application factory.

``create_app`` takes every collaborator as an optional argument so tests can
inject a temporary registry, and production can call ``build_app()`` with no
arguments and get environment-driven configuration.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mlservice import __version__
from mlservice.config.observability import ObservabilityConfig
from mlservice.config.service import ServiceConfig
from mlservice.monitoring.reporter import DriftReporter
from mlservice.observability.logging import configure_logging
from mlservice.observability.tracing import configure_tracing, instrument_app, shutdown_tracing
from mlservice.registry.base import ModelRegistry
from mlservice.registry.local import LocalFilesystemRegistry
from mlservice.serving.errors import register_exception_handlers
from mlservice.serving.middleware import RequestContextMiddleware
from mlservice.serving.model_holder import ModelHolder
from mlservice.serving.routes import build_metrics_router, router

logger = logging.getLogger(__name__)

DESCRIPTION = """
Inference service for a versioned model artifact.

The model, its preprocessor, its feature schema, its decision threshold, and its
provenance all come from one immutable registry artifact. Nothing about feature
handling is reimplemented here, which is what keeps training and serving in sync.
"""


def create_app(
    *,
    service_config: ServiceConfig | None = None,
    observability_config: ObservabilityConfig | None = None,
    registry: ModelRegistry | None = None,
    drift_reporter: DriftReporter | None = None,
    configure_logs: bool = True,
) -> FastAPI:
    """Build the application. Startup work happens in the lifespan handler."""
    service = service_config or ServiceConfig()
    observability = observability_config or ObservabilityConfig()
    model_registry = registry or LocalFilesystemRegistry(service.registry_root)
    reporter = drift_reporter or DriftReporter.from_config(observability)

    if configure_logs:
        configure_logging(observability)

    holder = ModelHolder(model_registry, service)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_tracing(observability, service.app_name)
        instrument_app(app)

        if service.fail_fast_on_missing_model:
            # Crash loudly at startup rather than serving 503s forever: a
            # missing model is a deploy bug, and the deploy should fail.
            holder.load()
        elif not holder.try_load():
            logger.warning(
                "starting without a model; /ready will report not_ready",
                extra={"load_error": holder.load_error},
            )

        try:
            yield
        finally:
            reporter.shutdown()
            shutdown_tracing()

    app = FastAPI(
        title=service.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        # Docs are on in every environment: an internal service whose schema is
        # undiscoverable costs more in support than it saves in obscurity.
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.state.service_config = service
    app.state.observability_config = observability
    app.state.registry = model_registry
    app.state.model_holder = holder
    app.state.drift_reporter = reporter

    app.add_middleware(RequestContextMiddleware, metrics_path=observability.metrics_path)
    register_exception_handlers(app)
    app.include_router(router)
    if observability.metrics_enabled:
        app.include_router(build_metrics_router(observability.metrics_path))

    return app


def build_app() -> FastAPI:
    """Entry point for ``uvicorn mlservice.serving.app:build_app --factory``."""
    return create_app()


#: Module-level app for ``uvicorn mlservice.serving.app:app``. Constructed lazily
#: by uvicorn's import, so importing this module in tests does not build one.
def __getattr__(name: str) -> FastAPI:
    if name == "app":
        return build_app()
    raise AttributeError(name)
