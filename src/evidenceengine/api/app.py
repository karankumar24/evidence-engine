"""FastAPI application factory."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
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
    "script-src 'self' 'unsafe-eval'; "  # Alpine 3 uses new Function() for expressions
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
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
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

    # Increase AnyIO thread pool from default 40 → 100.
    # Each LLM call in asyncio.to_thread occupies one slot for up to 30s (fallback timeout).
    # Without this, concurrent pipeline requests exhaust the pool and queue behind each other.
    import anyio
    anyio.to_thread.current_default_thread_limiter().total_tokens = 100
    if settings.sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            sentry_sdk.init(dsn=settings.sentry_dsn, integrations=[FastApiIntegration()])
            logger.info("Sentry initialized")
        except ImportError:
            logger.warning("sentry-sdk not installed; SENTRY_DSN ignored")

    # Pre-warm the openai module in a background thread. On macOS Tahoe the
    # first `from openai import AsyncOpenAI` triggers syspolicyd Gatekeeper
    # scanning of httpx/anyio/pydantic_core .so files (5-20 min, zero CPU).
    # Without this prewarm, the hang lands on the user's first pipeline run
    # mid-extract, with no visible feedback. Here it runs concurrently with
    # uvicorn startup and logs progress so the delay is observable.
    async def _prewarm_openai() -> None:
        import asyncio as _asyncio
        import time as _time
        def _do_import() -> float:
            t0 = _time.monotonic()
            from openai import OpenAI  # noqa: F401
            return _time.monotonic() - t0
        try:
            dur = await _asyncio.to_thread(_do_import)
            logger.info("openai module prewarmed in %.1fs", dur)
        except Exception:
            logger.exception("openai prewarm failed — first pipeline run may hang")
    app.state.openai_prewarm = asyncio.create_task(_prewarm_openai())

    # Pre-warm the TLS/SSL stack by making a synchronous HTTPS connection in a
    # background thread. On macOS Tahoe, the first TLS handshake triggers
    # Gatekeeper/keychain scanning that BLOCKS the asyncio event loop for several
    # minutes — making the server appear hung during the first LLM pipeline call.
    # Running a synchronous urllib request in a thread here warms the macOS
    # security subsystem so subsequent async httpx connections are non-blocking.
    async def _prewarm_tls() -> None:
        import time as _time
        import asyncio as _asyncio
        def _do_connect() -> float:
            import urllib.request
            t0 = _time.monotonic()
            try:
                urllib.request.urlopen(
                    (settings.llm_base_url or "https://openrouter.ai/api/v1") + "/models",
                    timeout=20,
                )
            except Exception:
                pass  # 401/404/network error is fine — the TLS handshake is what matters
            return _time.monotonic() - t0
        try:
            dur = await _asyncio.to_thread(_do_connect)
            logger.info("TLS stack prewarmed to LLM endpoint in %.1fs", dur)
        except Exception:
            logger.exception("TLS prewarm failed — first LLM pipeline call may stall")
    app.state.tls_prewarm = asyncio.create_task(_prewarm_tls())

    # Recover orphaned in-flight runs from a previous server crash/restart.
    # Any run still in a pipeline stage means its background task was killed
    # mid-flight — _mark_failed was never called. Mark them failed now so the
    # dashboard doesn't show perpetually-spinning EXTRACTING entries.
    # Wrapped in asyncio.timeout so a slow/unreachable DB never blocks startup.
    try:
        from datetime import datetime, timezone
        from sqlalchemy import select
        from evidenceengine.core.database import async_session_factory
        from evidenceengine.models.run import RunVersion
        _IN_FLIGHT = ("queued", "parsing", "extracting", "retrieving", "classifying")
        async with asyncio.timeout(15):
            async with async_session_factory() as _s:
                result = await _s.execute(
                    select(RunVersion).where(RunVersion.status.in_(_IN_FLIGHT))
                )
                orphans = result.scalars().all()
                if orphans:
                    for run in orphans:
                        run.status = "failed"
                        run.error_summary = "ServerRestart: pipeline was interrupted by server shutdown"
                        run.completed_at = datetime.now(timezone.utc)
                    await _s.commit()
                    logger.warning(
                        "Recovered %d orphaned in-flight run(s) → marked as failed",
                        len(orphans),
                    )
    except TimeoutError:
        logger.warning("DB unreachable at startup (15s timeout) — orphan recovery skipped, server still starting")
    except Exception:
        logger.exception("Failed to recover orphaned runs at startup — continuing")

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
        allow_headers=["Content-Type", "Authorization", "HX-Request", "HX-Target", "HX-Trigger", "HX-Vals"],
    )

    # Security headers on every response
    app.add_middleware(SecurityHeadersMiddleware)

    register_exception_handlers(app)

    from fastapi.staticfiles import StaticFiles

    STATIC_DIR = Path(__file__).parent.parent / "static"
    STATIC_DIR.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    from evidenceengine.api.routes.runs import router as runs_router
    from evidenceengine.api.routes.dashboard import router as dashboard_router
    from evidenceengine.api.routes.design_system import router as design_system_router
    from evidenceengine.api.routes.packets import router as packets_router
    from evidenceengine.api.routes.extraction import router as extraction_router
    from evidenceengine.api.routes.retrieval import router as retrieval_router
    from evidenceengine.api.routes.classification import router as classification_router
    from evidenceengine.api.routes.pipeline import router as pipeline_router
    from evidenceengine.api.routes.submit import router as submit_router

    app.include_router(runs_router)
    app.include_router(dashboard_router)
    app.include_router(design_system_router)
    app.include_router(packets_router)
    app.include_router(extraction_router)
    app.include_router(retrieval_router)
    app.include_router(classification_router)
    app.include_router(pipeline_router)
    app.include_router(submit_router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/dashboard", status_code=302)

    # ── Health endpoints ──────────────────────────────────────────────────────

    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        return {"status": "ok"}

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict:
        from sqlalchemy import text
        from evidenceengine.core.database import async_session_factory
        try:
            async with async_session_factory() as session:
                await session.execute(text("SELECT 1"))
            return {"status": "ok", "db": "reachable"}
        except Exception as exc:
            logger.error("Health check DB ping failed: %s", exc)
            return JSONResponse(
                status_code=503,
                content={"status": "error", "db": "unreachable"},
            )

    return app


app = create_app()
