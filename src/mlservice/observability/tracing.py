"""Optional OpenTelemetry tracing.

Tracing is an extra (`pip install '.[otel]'`). Every function here degrades to a
no-op when the packages are absent or when ``tracing_enabled`` is False, so the
import graph of the service does not depend on OpenTelemetry being installed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from mlservice.config.observability import ObservabilityConfig

logger = logging.getLogger(__name__)

_tracer: Any | None = None
_enabled = False


def configure_tracing(config: ObservabilityConfig, service_name: str) -> bool:
    """Install an OTLP tracer. Returns True when tracing actually came up."""
    global _tracer, _enabled

    if not config.tracing_enabled:
        _enabled = False
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio
    except ImportError:
        # Enabled but not installed is a configuration mistake worth shouting
        # about — yet not worth failing a deploy over.
        logger.warning(
            "tracing_enabled=True but OpenTelemetry is not installed; "
            "install the 'otel' extra. Continuing without tracing."
        )
        _enabled = False
        return False

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": config.service_namespace,
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBasedTraceIdRatio(config.trace_sample_ratio),
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=config.otlp_endpoint)))
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer("mlservice")
    _enabled = True
    logger.info("tracing enabled, exporting to %s", config.otlp_endpoint)
    return True


def instrument_app(app: Any) -> None:
    """Attach FastAPI auto-instrumentation when tracing is live."""
    if not _enabled:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        return
    FastAPIInstrumentor.instrument_app(app)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Start a span, or do nothing at all when tracing is off."""
    if not _enabled or _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as active:
        for key, value in attributes.items():
            active.set_attribute(key, value)
        yield


def is_enabled() -> bool:
    return _enabled


def shutdown_tracing() -> None:
    """Flush pending spans at process exit."""
    if not _enabled:
        return
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except ImportError:
        pass
