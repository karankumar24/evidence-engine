"""FastAPI application factory."""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from evidenceengine.core.config import settings
from evidenceengine.schemas.common import APIError

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_error_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _wants_html(request: Request) -> bool:
    """True when the client asked for HTML and the path is not a JSON API route."""
    if request.url.path.startswith("/api/"):
        return False
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers that emit a consistent error envelope.

    All error responses from Phase 5+ routes use:
        {"error": {"code": "...", "message": "...", "detail": ...}}

    Existing packets.py / extraction.py routes that raise HTTPException with a
    dict ``detail`` will surface as HTTP_{status_code} with the dict as the
    message string — acceptable for Phase 5. A full packets.py migration to
    APIError is out of scope here.
    """

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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="EvidenceEngine",
        version="0.1.0",
        description="Provenance-aware document verification system",
    )

    # CORS — allow all origins for dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register global exception handlers (Phase 5+)
    register_exception_handlers(app)

    # Mount static files for dashboard CSS/JS assets
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path

    STATIC_DIR = Path(__file__).parent.parent / "static"
    STATIC_DIR.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include routers
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

    @app.on_event("startup")
    async def startup_event() -> None:
        """Create upload directory on startup."""
        os.makedirs(settings.upload_dir, exist_ok=True)

    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
