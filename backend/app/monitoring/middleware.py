"""Request timing middleware: logs every API call and persists it for error-rate metrics."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger
from app.database.session import session_scope
from app.monitoring.metrics_service import record_request

logger = get_logger("app.request")

_SKIP_PREFIXES = ("/docs", "/redoc", "/openapi.json", "/favicon")
_POLLING = {"/api/v1/metrics", "/api/v1/documents", "/api/v1/dashboard", "/api/v1/evaluation/results"}


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            duration = (time.perf_counter() - start) * 1000
            path = request.url.path
            if not path.startswith(_SKIP_PREFIXES):
                logger.info("%s %s -> %d (%.1f ms) [%s]", request.method, path, status_code, duration, request_id)
                # Successful dashboard polling GETs are excluded to keep the table meaningful.
                polling = request.method == "GET" and status_code < 400 and path.rstrip("/") in _POLLING
                if path.startswith("/api/") and not polling:
                    try:
                        with session_scope() as db:
                            record_request(db, request.method, _normalise(path), status_code, duration)
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to persist request log", exc_info=True)


def _normalise(path: str) -> str:
    """Replace ids with placeholders so paths aggregate cleanly."""
    parts = []
    for p in path.split("/"):
        parts.append("{id}" if len(p) == 32 and all(c in "0123456789abcdef" for c in p) else p)
    return "/".join(parts)[:255]
