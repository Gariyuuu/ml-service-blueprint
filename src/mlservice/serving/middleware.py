"""HTTP middleware: request identity, access logs, and request metrics."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from mlservice.observability import metrics
from mlservice.observability.context import new_request_id, set_request_id

logger = logging.getLogger("mlservice.access")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, times the request, and emits one access log line."""

    def __init__(self, app: Callable[..., Awaitable[None]], *, metrics_path: str) -> None:
        super().__init__(app)
        self.metrics_path = metrics_path

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        set_request_id(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Exception handlers below convert this into a 500 response; record
            # the failure here so the counter is not lost.
            duration = time.perf_counter() - started
            route = self._route_label(request)
            metrics.record_request(request.method, route, 500)
            metrics.record_error(route, "unhandled")
            metrics.REQUEST_LATENCY.labels(method=request.method, route=route).observe(duration)
            logger.exception("request failed", extra={"method": request.method, "route": route})
            raise

        duration = time.perf_counter() - started
        route = self._route_label(request)
        response.headers[REQUEST_ID_HEADER] = request_id

        # Scraping /metrics should not itself dominate the metrics.
        if route != self.metrics_path:
            metrics.record_request(request.method, route, response.status_code)
            metrics.REQUEST_LATENCY.labels(method=request.method, route=route).observe(duration)
            if response.status_code >= 500:
                metrics.record_error(route, "server_error")
            elif response.status_code >= 400:
                metrics.record_error(route, "client_error")

            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "route": route,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(duration * 1000, 3),
                },
            )
        return response

    @staticmethod
    def _route_label(request: Request) -> str:
        """Use the *route template*, never the raw path.

        A metric labelled with raw paths grows one timeseries per distinct URL,
        which is how a metrics backend gets taken down by a scanner.
        """
        route = request.scope.get("route")
        path = getattr(route, "path", None)
        return str(path) if path else "unmatched"
