"""FastAPI application factory."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp

from evidenceengine.core.config import settings
from evidenceengine.schemas.common import APIError

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_error_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

APP_VERSION = "0.1.0"


# ── Security headers middleware ───────────────────────────────────────────────

_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: ASGIApp) -> StarletteResponse:  # type: ignore[override]
        response: StarletteResponse = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = _CSP
        return response


# ── Exception handlers ────────────────────────────────────────────────────────

def _wants_html(request: Request) -> bool:
    if request.url.path.startswith("/api/"):
        return False
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> Response:
        if _wants_html(request) and exc.status_code in (404, 500):
            template = "errors/404.html" if exc.status_code == 404 else "errors/500.html"
            return _error_templates.TemplateResponse(
                request, template, {"detail": str(exc.detail)}, status_code=exc.status_code
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": str(exc.detail),
                    "detail": None,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "detail": exc.errors(),
                }
            },
        )

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "detail": exc.detail,
                }
            },
        )


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from evidenceengine.core.logging_config import configure_logging
    configure_logging(debug=settings.debug)
    os.makedirs(settings.upload_dir, exist_ok=True)
    if settings.sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            sentry_sdk.init(dsn=settings.sentry_dsn, integrations=[FastApiIntegration()])
            logger.info("Sentry initialized")
        except ImportError:
            logger.warning("sentry-sdk not installed; SENTRY_DSN ignored")
    yield


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="EvidenceEngine",
        version=APP_VERSION,
        description="Provenance-aware document verification system",
        lifespan=lifespan,
    )

    # CORS — env-configured allow-list; credentials only when origins are explicit
    allow_credentials = "*" not in settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Security headers on every response
    app.add_middleware(SecurityHeadersMiddleware)

    register_exception_handlers(app)

    from fastapi.staticfiles import StaticFiles

    STATIC_DIR = Path(__file__).parent.parent / "static"
    STATIC_DIR.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    from evidenceengine.api.routes.packets import router as packets_router
    from evidenceengine.api.routes.extraction import router as extraction_router
    from evidenceengine.api.routes.retrieval import router as retrieval_router
    from evidenceengine.api.routes.classification import router as classification_router
    from evidenceengine.api.routes.pipeline import router as pipeline_router
    from evidenceengine.api.routes.runs import router as runs_router
    from evidenceengine.api.routes.dashboard import router as dashboard_router
    from evidenceengine.api.routes.design_system import router as design_system_router

    app.include_router(packets_router)
    app.include_router(extraction_router)
    app.include_router(retrieval_router)
    app.include_router(classification_router)
    app.include_router(pipeline_router)
    app.include_router(runs_router)
    app.include_router(dashboard_router)
    app.include_router(design_system_router)

    # ── Health endpoints ──────────────────────────────────────────────────────

    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        return {"status": "ok"}

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict:
        from sqlalchemy import text
        from evidenceengine.core.database import async_session_factory
        from fastapi import HTTPException as FastHTTPException
        try:
            async with async_session_factory() as session:
                await session.execute(text("SELECT 1"))
            return {"status": "ok", "db": "reachable"}
        except Exception as exc:
            logger.error("Health check DB ping failed: %s", exc)
            raise FastHTTPException(status_code=503, detail="Database unavailable")

    return app


app = create_app()
